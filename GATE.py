import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd

class GATE(nn.Module):

    def __init__(self, input_size, hidden_size, output_dim, kernel_size,
                 hidden_dropout_prob,
                 gate_dropout_prob,
                 num_channels,
                 norm_type='instance'):
        super(GATE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.norm_type = norm_type
        
        self.cnn = nn.Conv1d(1, num_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        
        if norm_type == 'batch':
            self.cnn_norm = nn.BatchNorm1d(num_channels)
        elif norm_type == 'group':
            num_groups = min(2, num_channels)
            self.cnn_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        elif norm_type == 'layer':
            self.cnn_norm = nn.GroupNorm(num_groups=1, num_channels=num_channels)
        elif norm_type == 'instance':
            self.cnn_norm = nn.InstanceNorm1d(num_channels, affine=True)
        else:
            self.cnn_norm = nn.Identity()
        
        self.activation = nn.GELU()
        
        self.channel_fusion = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(num_channels, max(1, num_channels // 2), 1),
            nn.ReLU(),
            nn.Conv1d(max(1, num_channels // 2), num_channels, 1),
            nn.Sigmoid()
        )
        
        self.path_gate = nn.Linear(input_size, hidden_size)
        self.path_feature = nn.Linear(input_size, hidden_size)

        self.expand_layer = nn.Linear(hidden_size, input_size)

        self.gate_dropout = nn.Dropout(gate_dropout_prob)
        self.feature_dropout = nn.Dropout(hidden_dropout_prob)

        self.layer_norm = nn.LayerNorm(input_size, eps=1e-12)
        self.output_projection = nn.Linear(input_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_tensor, return_gate_weights=False, return_attention_maps=False):
        input_expanded = input_tensor.unsqueeze(1)
        multi_channel_features = self.cnn(input_expanded)

        multi_channel_features = self.cnn_norm(multi_channel_features)
        multi_channel_features = self.activation(multi_channel_features)

        channel_weights = self.channel_attention(multi_channel_features)
        weighted_features = multi_channel_features * channel_weights

        cnn_output = self.channel_fusion(weighted_features).squeeze(1)

        feature_stream = self.path_feature(cnn_output)
        gate_signal = self.path_gate(cnn_output)
        selective_gate = torch.sigmoid(gate_signal)
        selective_gate = self.gate_dropout(selective_gate)
        gated_feature = feature_stream * selective_gate

        expanded_features = self.expand_layer(gated_feature)
        expanded_features = self.feature_dropout(expanded_features)

        residual_input = expanded_features + cnn_output

        residual_output = self.layer_norm(residual_input)

        activated_output = self.relu(residual_output)
        final_output = self.output_projection(activated_output)

        if return_attention_maps and return_gate_weights:
            return final_output, selective_gate, channel_weights
        elif return_gate_weights:
            return final_output, selective_gate
        elif return_attention_maps:
            return final_output, channel_weights
        else:
            return final_output


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def run_cross_validation(data, label, cv, learning_rate, batch_size, hidden_dim,
                         output_dim, kernel_size, patience, DEVICE,
                         hidden_dropout_prob, gate_dropout_prob, repeat, result_txt,
                         lambda_l1=0.0005, num_channels=8, norm_type='instance', input_size=None):
    all_results = []
    all_repeats_gate_weights = []
    for repeat_idx in range(repeat):
        print(f'==== Repeat {repeat_idx + 1} ====')
        fold_results = []
        current_repeat_fold_weights = []
        cv_iterator = KFold(n_splits=5, shuffle=True, random_state=42 + repeat_idx)
        for fold, (train_idx, test_idx) in enumerate(cv_iterator.split(data)):
            x_train, x_test = data[train_idx], data[test_idx]
            y_train, y_test = label[train_idx], label[test_idx]

            model = GATE(
                input_size=input_size,
                hidden_size=hidden_dim,
                output_dim=output_dim,
                kernel_size=kernel_size,
                hidden_dropout_prob=hidden_dropout_prob,
                gate_dropout_prob=gate_dropout_prob,
                num_channels=num_channels,
                norm_type=norm_type,
            ).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_function = torch.nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            x_train_tensor = torch.from_numpy(x_train).float().to(DEVICE)
            y_train_tensor = torch.from_numpy(y_train).float().to(DEVICE)
            x_test_tensor = torch.from_numpy(x_test).float().to(DEVICE)
            y_test_tensor = torch.from_numpy(y_test).float().to(DEVICE)

            train_data = TensorDataset(x_train_tensor, y_train_tensor)
            test_data = TensorDataset(x_test_tensor, y_test_tensor)

            early_stopping = EarlyStopping(patience=patience)
            best_corr_coef = -float('inf')

            train_loader = DataLoader(train_data, batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size, shuffle=False)

            for epoch in range(500):
                model.train()
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()

                    y_pred, gate_weights = model(x_batch, return_gate_weights=True)

                    mse_loss = loss_function(y_pred, y_batch.reshape(-1, 1))

                    l1_penalty = torch.sum(torch.abs(gate_weights))

                    total_loss = mse_loss + lambda_l1 * l1_penalty

                    total_loss.backward()
                    optimizer.step()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                model.eval()
                y_test_preds, y_test_trues = [], []
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        y_test_pred = model(x_batch)
                        y_test_preds.extend(y_test_pred.cpu().numpy().reshape(-1).tolist())
                        y_test_trues.extend(y_batch.cpu().numpy().reshape(-1).tolist())

                if np.std(y_test_trues) == 0 or np.std(y_test_preds) == 0:
                    corr_coef = 0
                else:
                    corr_coef = np.corrcoef(y_test_preds, y_test_trues)[0, 1]

                scheduler.step(-corr_coef)

                if corr_coef > best_corr_coef:
                    best_corr_coef = corr_coef

                early_stopping(corr_coef)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            model.eval()
            with torch.no_grad():
                if hasattr(model, 'path_feature') and isinstance(model.path_feature, torch.nn.Linear):
                    static_weights = model.path_feature.weight.data.cpu()

                    all_gate_weights = []
                    for x_batch, _ in test_loader:
                        _, gate_values = model(x_batch, return_gate_weights=True)
                        all_gate_weights.append(gate_values.cpu())

                    all_gate_weights_tensor = torch.cat(all_gate_weights, dim=0)

                    avg_gate_values = torch.mean(all_gate_weights_tensor, dim=0)
                    effective_weights = avg_gate_values.unsqueeze(1) * static_weights
                    avg_weights_for_fold = torch.mean(torch.abs(effective_weights), dim=0).numpy()

                    current_repeat_fold_weights.append(avg_weights_for_fold)

            fold_results.append(best_corr_coef)
            print(f'Repeat {repeat_idx + 1} Fold {fold + 1}: PCC={best_corr_coef:.4f}')
        all_results.extend(fold_results)

        if current_repeat_fold_weights:
            avg_weights_for_repeat = np.mean(current_repeat_fold_weights, axis=0)
            all_repeats_gate_weights.append(avg_weights_for_repeat)

    with open(result_txt, 'w', encoding='utf-8') as f:
        f.write('repeat\tfold\tPCC\n')
        for i, corr in enumerate(all_results):
            repeat_idx = i // 5 + 1
            fold_idx = i % 5 + 1
            f.write(f'{repeat_idx}\t{fold_idx}\t{corr:.6f}\n')
        arr = np.array(all_results)
        f.write('\nmean_PCC\n')
        f.write(f'{arr.mean():.6f}\n')
    print('Results written to', result_txt)

    if all_repeats_gate_weights:
        final_feature_importance = np.mean(all_repeats_gate_weights, axis=0)
    else:
        final_feature_importance = None

    return final_feature_importance


def predict(model, data, device):
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GATE Model for training and prediction.")

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for CNN')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout probability for hidden layers')
    parser.add_argument('--gate_dropout_prob', type=float, default=0.1, help='Dropout probability for gate mechanism')
    parser.add_argument('--num_channels', type=int, default=8, help='Number of channels in CNN')
    parser.add_argument('--norm_type', type=str, default='instance', choices=['batch', 'group', 'layer', 'instance', 'none'], help='Normalization type')

    # Training parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], help='Mode of operation: train or predict')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default='GATE_model.pth', help='Path to save/load model')
    parser.add_argument('--data_path', type=str, help='Path to input data (for training or prediction)')
    parser.add_argument('--pheno_path', type=str, help='Path to pheno(for training)')
    parser.add_argument('--trait_column', type=str, default='0', help='Column index (0-based) or name of the trait to predict from pheno_path.Default is the first column.')
    parser.add_argument('--lambda_l1', type=float, default=0.0005, help='L1 regularization strength for gate weights')
    parser.add_argument('--repeat', type=int, default=5, help='Number of times to repeat the cross-validation')
    parser.add_argument('--result_txt', type=str, default='results.txt', help='Path to save training results')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        # Load data
        transcriptome = pd.read_csv(args.data_path, index_col=0)
        pheno = pd.read_csv(args.pheno_path, index_col=0)

        common_index = transcriptome.index.intersection(pheno.index)
        transcriptome = transcriptome.loc[common_index]
        pheno = pheno.loc[common_index]

        zero_ratio = (transcriptome == 0).sum(axis=0) / transcriptome.shape[0]
        transcriptome = transcriptome.loc[:, zero_ratio < 0.8]
        data = transcriptome.values
        
        try:
            trait_column = int(args.trait_column)
            label = pheno.iloc[:, trait_column].values.reshape(-1, 1)
        except ValueError:
            if args.trait_column in pheno.columns:
                label = pheno[args.trait_column].values.reshape(-1, 1)
            else:
                print(f"Error: Trait column '{args.trait_column}' not found in labels file.")
                import sys
                sys.exit(1)

        input_size = data.shape[1] 
        gene_names = transcriptome.columns

        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

        feature_importances = run_cross_validation(
            data=data,
            label=label,
            cv=cv_splitter,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_size,
            output_dim=1,
            kernel_size=args.kernel_size,
            patience=30,
            DEVICE=device,
            hidden_dropout_prob=args.hidden_dropout_prob,
            gate_dropout_prob=args.gate_dropout_prob,
            repeat=args.repeat,
            result_txt=args.result_txt,
            lambda_l1=args.lambda_l1,
            num_channels=args.num_channels,
            norm_type=args.norm_type,
            input_size=input_size
        )

        if feature_importances is not None:
            importance_df = pd.DataFrame({
                'gene': gene_names,
                'importance': feature_importances.flatten()
            })
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            importance_filename = args.result_txt.replace('.txt', '_gene_importance_scores.csv')
            importance_df.to_csv(importance_filename, index=False, encoding='utf-8')
            print(f"Gene importance scores saved to {importance_filename}")

        print("Training process completed successfully.")

        # Train a final model on all available data
        print("\n==== Training final model on all data ====")
        final_model = GATE(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_dim=1,
            kernel_size=args.kernel_size,
            hidden_dropout_prob=args.hidden_dropout_prob,
            gate_dropout_prob=args.gate_dropout_prob,
            num_channels=args.num_channels,
            norm_type=args.norm_type,
        ).to(device)

        # Scale the full data
        scaler_final = StandardScaler()
        data_scaled_final = scaler_final.fit_transform(data)
        x_full_tensor = torch.from_numpy(data_scaled_final).float().to(device)
        y_full_tensor = torch.from_numpy(label).float().to(device)
        
        full_dataset = TensorDataset(x_full_tensor, y_full_tensor)
        full_data_loader = DataLoader(full_dataset, args.batch_size, shuffle=True)

        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
        final_loss_function = torch.nn.MSELoss()

        best_loss = float('inf')
        best_model_state = None

        for epoch_final in range(args.epochs):
            final_model.train()
            epoch_losses = []
            for x_batch_full, y_batch_full in full_data_loader:
                final_optimizer.zero_grad()
                y_pred_full, gate_weights_full = final_model(x_batch_full, return_gate_weights=True)
                mse_loss_full = final_loss_function(y_pred_full, y_batch_full.reshape(-1, 1))
                l1_penalty_full = torch.sum(torch.abs(gate_weights_full))
                total_loss_full = mse_loss_full + args.lambda_l1 * l1_penalty_full
                total_loss_full.backward()
                final_optimizer.step()
                epoch_losses.append(total_loss_full.item())

            avg_epoch_loss = np.mean(epoch_losses)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = final_model.state_dict().copy()

            if (epoch_final + 1) % 50 == 0 or epoch_final == args.epochs - 1:
                print(f"Final Model Epoch {epoch_final + 1}/{args.epochs}, Loss: {avg_epoch_loss:.4f}, Best Loss: {best_loss:.4f}")

        if args.model_path and best_model_state is not None:
            torch.save(best_model_state, args.model_path)
            print(f"Final model (best loss: {best_loss:.4f}) saved to {args.model_path}")

    elif args.mode == 'predict':
        predict_df = pd.read_csv(args.data_path, index_col=0)
        sample_names = predict_df.index.tolist()
        input_size = predict_df.shape[1]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(predict_df.values)
        data = torch.from_numpy(scaled_data).float()

        if not args.model_path:
            print("Error: '--model_path' is required for prediction mode.")
            import sys
            sys.exit(1)

        model = GATE(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_dim=1,
            kernel_size=args.kernel_size,
            hidden_dropout_prob=args.hidden_dropout_prob,
            gate_dropout_prob=args.gate_dropout_prob,
            num_channels=args.num_channels,
            norm_type=args.norm_type,
        ).to(device)
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded from {args.model_path}")

        predictions = predict(model, data, device)
        predictions_np = predictions.cpu().numpy().flatten()
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Sample': sample_names,
            'Prediction': predictions_np
        })
        
        # Save predictions to CSV
        output_file = args.data_path.replace('.csv', '_predictions.csv')
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nPredictions saved to {output_file}")
        print(f"\nFirst 10 predictions:")
        print(results_df.head(10).to_string(index=False)) 
