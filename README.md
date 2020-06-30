# Face-Mask-detector

Il codice è stato preso dal seguente link: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/ (ho cambiato solo la parte del face detector che non funzionava benissimo, ora uso facenet_pytorch).

* **train_mask_detector.py** training su dataset con seguente struttura: *with_mask* - volti con maschera, *without_mask* - volti senza maschera. *python train_mask_detector.py --dataset dataset*
* **detect_mask_image.py** individua i volti con e senza mascherina all'interno dell'immagine. *python detect_mask_image.py --image "path to image"*
  
Nel repo ho anche aggiunto "mask_detector.model" che sono i pesi ottenuti da training sul famoso "dataset sintetico".

In **requirements.txt** ho i pacchetti che avevo installato nel mio conda env, è probabile che non sia pulitissimo. In ogni caso dando il comando "conda create --name <env> --file <this file>" dovrebbe andare tutto senza problemi. Ho usato Python 3.7.7
