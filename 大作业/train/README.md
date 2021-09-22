AFEW-VA dataset
===============

Organisation
------------
In the data folder are 600 folders, one for each video.
Each video folder contains images (the frames of the video) from 00000.png to 00NNN.png as well as a .json file containing the annotations (valence, arousal and facial landmarks).

The json file are organised as follows:
```json
{
  "actor": "Actor Name",
  "frames": {
    "00000": {
      "arousal": arousal_value,
      "landmarks": [[x_1, y_1], [x_2, y_2], ..., [x_68, y_68]],
      "valence": 0.0
    },
    "00001": {
      "arousal": arousal_value,
      "landmarks": [[x_1, y_1], [x_2, y_2], ..., [x_68, y_68]],
      "valence": 0.0
    },

    ...,

    "00NNN": {
      "arousal": arousal_value,
      "landmarks": [[x_1, y_1], [x_2, y_2], ..., [x_68, y_68]],
      "valence": 0.0
    },
  },
  "video_id": "video_id"
}
```

Valence and arousal are signed integers between -10 and 10, landmarks are 2-dimensional arrays of 68 facial landmarks, defined by their position (x, y).


Citing
------
If you use the AFEW-VA dataset please cite the following papers:

* J. Kossaifi, G. Tzimiropoulos, S. Todorovic and M. Pantic. AFEW-VA for valence and arousal estimation In-The-Wild. Image and Vision Computing, 2016 (submitted).
*  Abhinav Dhall, Roland Goecke, Simon Lucey, and Tom Gedeon. Static Facial Expressions in Tough Conditions: Data, Evaluation Protocol And Benchmark, First IEEE International Workshop on Benchmarking Facial Image Analysis Technologies BeFIT, IEEE International Conference on Computer Vision ICCV2011, Barcelona, Spain, 6-13 November 2011


Bibtex:
-------
```
@article{kossaifi_AFEWVA,
    author = {J. Kossaifi and G. Tzimiropoulos and S. Todorovic and M. Pantic},
    journal = {Image and Vision Computing},
    title = {AFEW-VA for valence and arousal estimation In-The-Wild, submitted},
    year = {2016},
}

@article{collection2012dhall,
 author = {Dhall, Abhinav and Goecke, Roland and Lucey, Simon and Gedeon, Tom},
 title = {Collecting Large, Richly Annotated Facial-Expression Databases from Movies},
 journal = {IEEE MultiMedia},
 issue_date = {July 2012},
 volume = {19},
 number = {3},
 month = jul,
 year = {2012},
 pages = {34--41},
}
```


