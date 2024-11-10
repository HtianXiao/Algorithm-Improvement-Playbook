import torch
import torch.nn.functional as F
import math

class ChirpletKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale_base=1.0,
        scale_chirplet=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        Initialize ChirpletKANLinear layer.

        Parameters:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            scale_base (float): Scaling factor for base weight initialization.
            scale_chirplet (float): Scaling factor for Chirplet coefficients initialization.
            base_activation (nn.Module): Activation function for the base linear component.
            use_bias (bool): Whether to use bias term.
        """
        super(ChirpletKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_base = scale_base
        self.scale_chirplet = scale_chirplet
        self.base_activation = base_activation()
        self.use_bias = use_bias

        # Initialize weights for base linear transform
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialize Chirplet parameters
        self.chirplet_weights = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = torch.nn.Parameter(torch.ones(out_features, in_features))
        self.translation = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.frequency = torch.nn.Parameter(torch.ones(out_features, in_features))

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            std = self.scale_chirplet / math.sqrt(self.in_features)
            self.chirplet_weights.uniform_(-std, std)

        torch.nn.init.ones_(self.scale)
        torch.nn.init.zeros_(self.translation)
        torch.nn.init.ones_(self.frequency)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def chirplet_transform(self, x):
        """
        Compute the Chirplet transform of the input x.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Chirplet transformed output of shape (batch_size, out_features).
        """
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(1)

        scale_expanded = self.scale.unsqueeze(0)
        translation_expanded = self.translation.unsqueeze(0)
        frequency_expanded = self.frequency.unsqueeze(0)

        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Compute Chirplet function
        chirplet_real = torch.cos(2 * math.pi * frequency_expanded * x_scaled)
        chirplet_gaussian = torch.exp(-0.5 * x_scaled ** 2)
        chirplet = chirplet_real * chirplet_gaussian

        chirplet_weights_expanded = self.chirplet_weights.unsqueeze(0)
        chirplet_output = (chirplet * chirplet_weights_expanded).sum(dim=2)

        return chirplet_output

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        chirplet_output = self.chirplet_transform(x)

        output = base_output + chirplet_output

        if self.use_bias:
            output += self.bias

        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        Calculate regularization loss for chirplet coefficients.

        Parameters:
            regularize_coeffs (float): Regularization coefficient.

        Returns:
            torch.Tensor: Regularization loss.
        """
        coeffs_l2 = self.chirplet_weights.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class ChirpletKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        scale_base=1.0,
        scale_chirplet=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        Initialize the ChirpletKAN model.

        Parameters:
            layers_hidden (list): List of layer input and output dimensions.
            scale_base (float): Scaling factor for base weight initialization.
            scale_chirplet (float): Scaling factor for Chirplet coefficients.
            base_activation (nn.Module): Activation function for base linear component.
            use_bias (bool): Whether to use bias term.
        """
        super(ChirpletKAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChirpletKANLinear(
                    in_features,
                    out_features,
                    scale_base=scale_base,
                    scale_chirplet=scale_chirplet,
                    base_activation=base_activation,
                    use_bias=use_bias,
                )
            )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        Calculate total regularization loss for the model.

        Parameters:
            regularize_coeffs (float): Regularization coefficient.

        Returns:
            float: Total regularization loss.
        """
        return sum(
            layer.regularization_loss(regularize_coeffs)
            for layer in self.layers
        )
