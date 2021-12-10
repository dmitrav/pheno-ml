
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize


class Autoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),  # 128 x 64 x 64

            torch.nn.Conv2d(128, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),  # 64 x 32 x 32

            torch.nn.Conv2d(64, 32, (3,3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            torch.nn.ReLU(True),

            torch.nn.Conv2d(32, 4, (3, 3), stride=(1,1), padding=(1,1)),  # 4 x 32 x 32
        )

        self.decoder = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(4, 32, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),  # 32 x 64 x 64

            torch.nn.ConvTranspose2d(32, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),  # 64 x 128 x 128

            torch.nn.ConvTranspose2d(64, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(128, 1, (3,3), stride=(1,1), padding=(1,1)),
            torch.nn.Sigmoid()
        )
        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def initialize_model(device):
    
    model = Autoencoder().to(device)
    # load a trained model to use it in the transform
    model.load_state_dict(torch.load('trained_model.torch', map_location=device))
    model.eval()
    return model


if __name__ == '__main__':

    device = torch.device('cpu')
    image_size = 128

    convae = initialize_model(device)
    preprocessor = lambda x: torch.unsqueeze(Resize(size=image_size)(x / 255.), 0)
    feature_extractor = lambda x: convae.encoder(preprocessor(x))  # feed tensor image
    reconstructor = lambda x: convae.decoder(x)  # feed extracted features

    # read example image
    image = read_image('HCT15_example.jpg')
    # extract features
    features = feature_extractor(image.to(device))
    numpy_features = features.reshape(-1).detach().numpy()
    # inspect reconstructions
    recs = reconstructor(features)
    numpy_recs = recs.detach().numpy()

