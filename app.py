from flask import Flask
from flask import request
from PIL import Image
import requests
import numpy as np
import torch
import random

app = Flask(__name__)


@app.route('/classify')
def classify():
    link = request.args.get('link')
    print(link)

    im = Image.open(requests.get(link, stream=True).raw)
    pix = np.array(im)
    input = torch.from_numpy(pix)
    print(input)
    return str(random.randint(1,5))


if __name__ == '__main__':
    app.run()
