# Datasets 

We provide the datasets used in our benchmarking study with links to their original sources (to avoid licensing issues).
If you are unable to run with the provided linked datasets, send an email (corresponding email in manuscript) for preprocessed versions. 

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">\textbf{corpus}</th>
    <th class="tg-c3ow">\textbf{task}</th>
    <th class="tg-c3ow">$D$</th>
    <th class="tg-c3ow">$K$</th>
    <th class="tg-c3ow">$I$</th>
    <th class="tg-c3ow">$W$</th>
    <th class="tg-c3ow">$V$</th>
    <th class="tg-c3ow">Link</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">20news</td>
    <td class="tg-c3ow">newswire topic</td>
    <td class="tg-c3ow">18,848</td>
    <td class="tg-c3ow">20</td>
    <td class="tg-c3ow">5e-4</td>
    <td class="tg-c3ow">240</td>
    <td class="tg-c3ow">212,267</td>
      <td>
        <a href="https://github.com/hendrycks/error-detection/tree/master/NLP/Categorization/data">Link</a>
      </td>
  </tr>
  <tr>
    <td class="tg-0pky">IMDB</td>
    <td class="tg-c3ow">movie review</td>
    <td class="tg-c3ow">348,415</td>
    <td class="tg-c3ow">10</td>
    <td class="tg-c3ow">0.03</td>
    <td class="tg-c3ow">325.6</td>
    <td class="tg-c3ow">115,073</td>
      <td>
        <a href="https://drive.google.com/drive/folders/1rASDy8v4QPq4ZNEZqIJo5dqcxAGINW8K">Link</a>
      </td>
  </tr>

  <tr>
    <td class="tg-0pky">CLINC-OOS</td>
    <td class="tg-c3ow">intent detection</td>
    <td class="tg-c3ow">22,500</td>
    <td class="tg-c3ow">150</td>
    <td class="tg-c3ow">0</td>
    <td class="tg-c3ow">8</td>
    <td class="tg-c3ow">6,188</td>
      <td>
        <a href="https://www.tensorflow.org/datasets/catalog/clinc_oos">Link</a>
      </td>
  </tr>
  <tr>
    <td class="tg-0pky">Reuters ApteMod</td>
    <td class="tg-c3ow">newswire topic</td>
    <td class="tg-c3ow">10,786</td>
    <td class="tg-c3ow">90</td>
    <td class="tg-c3ow">0.14</td>
    <td class="tg-c3ow">125.2</td>
    <td class="tg-c3ow">65,035</td>
      <td>
        <a href="http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html">Link</a>
      </td>
  </tr>
  <tr>
    <td class="tg-0pky">AAPD</td>
    <td class="tg-c3ow">academic paper subject</td>
    <td class="tg-c3ow">55,840</td>
    <td class="tg-c3ow">54</td>
    <td class="tg-c3ow">0.04</td>
    <td class="tg-c3ow">145.4</td>
    <td class="tg-c3ow">66,854</td>
      <td>
        <a href="https://git.uwaterloo.ca/jimmylin/Castor-data/tree/master/datasets/AAPD/data">Link</a>
      </td>    
  </tr>
  <tr>
    <td class="tg-0pky">Amazon Reviews (\#4)</td>
    <td class="tg-c3ow">product sentiment</td>
    <td class="tg-c3ow">8,000</td>
    <td class="tg-c3ow">2</td>
    <td class="tg-c3ow">0</td>
    <td class="tg-c3ow">189.3</td>
    <td class="tg-c3ow">21,514</td>
      <td>
        <a href="https://github.com/declare-lab/kingdom/tree/master/dataset">Link</a>
      </td>        
  </tr>
</tbody>
</table>
<sub>$D$ denotes the number of documents in the dataset, $K$ the number of classes, $I$ the class imbalance ratio, $W$ the average number of words per document, $V$ the total vocabulary size respectively.</sub>


## References

```
@article{lang199520,
author = "Ken Lang",
  title="Newsweeder: Learning to filter netnews. version 20news-18828",
  journal = "Machine Learning Proceedings 1995",
address = "San Francisco (CA)",
pages = "331 - 339",
year = "1995"
}

@article{hendrycks2016baseline,
  title={A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks},
  author={Hendrycks, Dan and Gimpel, Kevin},
  journal = {5th International Conference on Learning Representations},
  year={2017}
}
```
```
@inproceedings{diao_2014,
    title = {Jointly Modeling Aspects, Ratings and Sentiments for Movie Recommendation ({JMARS})},
  author={Diao, Qiming and Qiu, Minghui and Wu, Chao-Yuan and Smola, Alexander J and Jiang, Jing and Wang, Chong},
  booktitle={Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={193--202},
  year={2014}
}
```

```
@inproceedings{larson2019evaluation,
    title = "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
    author = "Larson, Stefan  and
      Mahendran, Anish  and
      Peper, Joseph J.  and
      Clarke, Christopher  and
      Lee, Andrew  and
      Hill, Parker  and
      Kummerfeld, Jonathan K.  and
      Leach, Kevin  and
      Laurenzano, Michael A.  and
      Tang, Lingjia  and
      Mars, Jason",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    pages = "1311--1316"
}
```

```
@article{APTE94,
 author = "Chidanand Apt{\'{e}} and Fred Damerau and Sholom M. Weiss",
 title = {Automated Learning of Decision Rules for Text Categorization},
 journal = "ACM Transactions on Information Systems",
 year = 1994
}
```

```
@inproceedings{yang2018sgm,
    title = "{SGM}: Sequence Generation Model for Multi-label Classification",
    author = "Yang, Pengcheng  and
      Sun, Xu  and
      Li, Wei  and
      Ma, Shuming  and
      Wu, Wei  and
      Wang, Houfeng",
    booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
    month = aug,
    year = "2018",
    pages = "3915--3926"
}
```
