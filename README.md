# Project of digit recognition
Project of 4th semester of MIPT.

## Software development plan
More information can be found in [SDP][]

## Tutorial of Nernet
Used files: [NerNet][] and [Mnist][]  
The lib have three architecture of neural networks:  
* Dense-800
* Dense-200 - Dense-100 - Dense-50
* Conv-32 - MaxPool - Dense-256

You should use the method **_make_and_check_** of the class **_NerNet_**. The parametr _path_ is responsible of architecture of nernet. For example, if _path = "Weight/Conv"_, class uses 3rd acrhitecture, moreover, if directory is empty, nernet will fit itself and save weights. If directory is full, nernet will use these weights.  

## Developers

* [Pavel][]    
* [Nikita][]

[SDP]: ./SDP/SDP.pdf
[NerNet]: ./Ver1.0/NerNet.py
[Mnist]: ./Ver1.0/mnist.py
[Pavel]: https://github.com/PaulZakharov
[Nikita]: https://github.com/Tismoney
