'''
from PIL import Image
import numpy as np

mask = np.array(Image.open("71619_sat_26_json/label.png"))
print(mask.shape)
print(np.unique(mask))
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

mask = np.array(Image.open("71619_sat_26_json/label.png"))

print(mask.dtype)
print(mask.shape)
print(np.unique(mask, return_counts=True))

plt.imshow(mask, cmap="gray")
plt.colorbar()
plt.show()