# MURA Ensemble Deep Neural Network & Convolution Neural Network Classifier

### Authors

Mithun Ghosh<br>

This repository contains a Deep Neural Network and Convolution Neural Network(CNN) based Ensemble model for classifying MURA humerus dataset.

Libraries used-
1. Keras<br>
2. Tensorflow<br>
3. Numpy<br>

<h3>Instructions for Running</h3>
1. Make sure you have the necessary libaries installed.<br>
2. Run the 'main.py' file as-<br>
'''
python3 main.py
'''



\\begin{document}
\\maketitle
\\section{Project Demand Evaluation Method}
For the bubble chart, we need the projected demand and the number of enrolled students in the $x$ and $y$ axis, respectively. The projected demand data can be collected from \href{https://www.onetonline.org/}{ONET}. The CIP taxonomy is organized on two levels: 1) the first two-digit series, 2) the four-digit series. The first two-digit series represent the most general groupings of related programs. The four-digit series represent intermediate groupings of programs that have comparable content and objectives. For example, the CIP code for the Computer Programmers is: 15-1131.00., where the digits ‘15’ represents ENGINEERING TECHNOLOGIES/TECHNICIANS, the last six digits ‘1131.00 represents program for the Computer Programmers. Based on the CIP code, the projected demand data for any program can be collected from \href{https://www.onetonline.org/}{ONET}. Say,
\\begin{align*}
\text{the annual median/mean wages}= W,\\
\text{the number of employments}=N,\\
\text{projected growth over the years in percentage} = i.
\end{align*}
We need to put all the information into the projectile demand formulation. Missing any of the values can be problematic in getting the true information for the projectile demand. For that, we come up with the following formulation.
\begin{align*}
\text{The current demand}, CD=W*N,\\
\text{The future demand},  FD=(1+i)*W*N\\
\text{The projectile demand}, PD=FD-CD=i*W*N.  
\\end{align*}

In this evaluation metrics, we include all the needed details to form the proper demand criteria. 
For example, we can consider the CIP code: 15-1131.00, which is for the Computer Programmers. From \href{https://www.onetonline.org/link/summary/15-1131.00}{ONET}, we get the following value: current median wage, W= $\$84290$, current employment, $N=250000$ and the projected growth over the years in percentage, $i=-2\%$.Thus, projectile demand, $PD=\$0.98\times84290\times250000= \$2.065\times10^{10}$
\newpage
For some programs, there may be multiple categories of demand. For example, the computer science program has demand in software engineering, hardware engineering, database administrator, etc. For these cases, we are using a weighted average of all the demand categories to compute the overall demand for the program.
Let, for the $t$-th demand category,
\begin{align*}
\text{the annual median/mean wages}= W_t,\\
\text{the number of employments}=N_t,\\
\text{projected growth over the years in percentage} = i_t.
\end{align*}
\begin{align*}
\text{The overall annual salary},
W=\frac{\sum_{t=1}^n N_t*W_t}{\sum_{t=1}^n N_t}\\
\text{The overall projected growth over the years in percentage},
i=\frac{\sum_{t=1}^n N_t*i_t}{\sum_{t=1}^n N_t}\\
\text{The average number of employee},\bar{N}=\frac{\sum_{t=1}^n N_t}{n}\\
\text{The current demand}, CD=\bar{N}*W,\\
\text{The future demand},  FD=(1+i)*\bar{N}*W\\
\text{The projectile demand}, PD=FD-CD=i*\bar{N}*W,
\end{align*}
where, $t=1,2,...,n$
\end{document}

