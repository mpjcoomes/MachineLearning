Machine Learning Variable Selection
===================================

The purpose of a model determines the standards guiding suitable variable selection. The purpose in credit scoring research is to construct models in a manner inferable to industry. Above all, selection is determined by what creates a parsimonious facsimile of a branch-lending model.

Lenders have *a priori* beliefs that demographic variables (e.g. age, sex) are predictive and must be included in credit scoring models (Thomas, Banasik, & Crook, 2001). These variables form a standardised set included in most logistic regression (LOG) credit scoring. In recent years, predictions have been improved by adding unique lender-constructed behavioural variables (BV). To prevent model parsimony and interpretability from being sacrificed, these new variables must be restricted to those that are most predictive.

Credit scoring literature contains precedents of both greedy and judicious approaches to variable selection (i.e. include all vs. remove non-predictors). Choice of approach is determined largely by how authors assess their models. Bucker, Kampen and Kramer (2013) used a greedy approach and maintain that the enormous size of credit data sets provides sufficient degrees of freedom to accommodate insignificant regressors. They emphasised economic significance, meaning degree of misspecification or predictive accuracy (Engsted, 2009), over statistical significance. In contrast, Jacobson and Roszbach (2003) used the judicious approach, including 17 variables and discarding 57. They cite insufficient univariate relationship as justification, along with the multicollinearity common to data sets where variables measure related constructs.

Some experimental data sets include hundreds of BVs. Applying greedy variable selection to such experimental data can lead to LOG models overloaded with hundreds of variables. This maximises explanatory scope at the cost of explanatory power (Marin & Robert, 2014, p.91). When parsimony and interpretability are the priority, a judicious approach including only the most predictive BVs is optimal.

There is rarely a theoretical justification to guide selection of the most predictive BVs. A preliminary LOG including all variables can produce assessments of variable importance via significance and effect size values. However, such assessments are unreliable due to the multicollinearity between related BVs and other variables. Experimental research needs an accurate means of assessing variable importance. Machine-learning variable selection meets this need.

Common variable assessment strategies in credit scoring include automated LOG variable selection (e.g. stepwise, forward and backward), Factor Analysis, Principal Component Analysis, and field-specific statistics such as 'Weight of Evidence' and 'Information Value' (Lin, 2013). However, the restriction to traditional approaches in application scoring (e.g. LOG) need not extend to the variable assessment portion of model construction. On this account, the most predictive BVs can be assessed with a recent machine-learning classification algorithm, which along with the *a priori* variables, can compose the final LOG model variables.

Variable selection is a process whereby a heuristic or algorithm identifies the variables that best accomplish a given modelling objective (e.g. explanatory value, prediction accuracy). The current apex of literature is 'feature selection' in genomics. Gene expression microarray data sets range from 20,000 to 60,000 variables (Dziuda, 2010, p.100; Guyon & Elisseeff, 2003, p.1158), often numeric, coded (i.e. not intuitively interpretable), and with only cursory theoretical knowledge available to guide selection. Strategies from this body of literature can be adapted to identify the most predictive BVs.

The three main strategies in feature selection are 'wrappers', 'filters' and 'embedded methods' (Guyon & Elisseeff, 2003). With wrappers, the variable importance measures of a supervised learning machine, trained to predict a response variable, are used to determine variable selection in subsequent models. Filters, in contrast, assess importance during a 'pre-processing step' separate from the response variable, while embedded methods are automated and self-contained within the model (e.g. stepwise selection in regression).

Sharma (2012) is an example of a wrapper applied to credit scoring. This author first created complex interaction terms from the available variables. Including every term in an application scoring LOG would introduce superfluous variables and cause over-fitting (Sharma, Overstreet, & Beling, 2009). Sharma used a Random Forest (RF) algorithm as wrapper, applied via the 'randomForest' R package (Liaw & Wiener, 2002), to identify which terms were predictive and warranted inclusion. RF is a Decision Trees (DT) based machine-learning technique. It creates a model that is robust to the distorting influences of complex variable interactions and interrelationships that would render a parametric model unreliable. Similar to Sharma, other credit scoring projects can leverage DT-based machine learners to select BVs.

DT is a supervised learning approach made up of 'decision nodes'. Each decision node contains an individual test function of discrete outcomes. DT organises these nodes in a recursive, unidirectional, hierarchical fashion by repeated application of the test function. Tree 'induction' (i.e. training) starts with all data set records at the 'root' node and corresponding test function. The function splits records into subsets that are input, via 'branches', to subordinate 'leaf' nodes, which in turn split records to lower nodes. The output label of a leaf node, which is a document class code or an average numeric point estimate, constitutes the DT prediction. 'Gini Impurity' is a traditional splitting criterion that assesses the homogeneity of records in a node and inducts Classification And Regression Trees (CART; Breiman, Friedman, Olshen, & Stone, 1984, p.30; p.109). Gini Impurity is the complement of the sum of squared probabilities of randomly selecting 'correct' data set records for that node (de Ville, 2006, p.105). More correctly classified records leads to a smaller probability complement and a proportionally smaller impurity value. Splitting repeats until all branches lead to 'pure' nodes. The number of nodes is restricted during induction or cut back from pure nodes (i.e. 'prepruning' and 'postpruning') to prevent variance from small case numbers and noise from compromising DT accuracy (Alpaydin, 2010, p.185-188; Larose, 2005, p.109-115).

In early DT application the technique was proposed as "a radical new method for analyzing survey data" (Morgan & Sonquist, 1963, p.415). Social scientists of that period started to mine new census data sets that included a greater number of records and variables. Popular statistical tools were classical parametric methods. Replete with assumptions and distribution restrictions, these methods are distorted by the intercorrelations and interactions in such data sets. DT is a robust nonparametric alternative. A "regression model could be masked by a combination of both interactive and multicollinear effects", writes de Ville (2006), while a DT "would be immune to the model-defeating characteristics of these effects, and would be a useful tool in identifying terms for the regression equation to help the models perform better" (p.31-32).

DT retains these useful statistical properties and today serves as an interpretable and fast-training nonparametric exploratory alternative (Wang, Ma, Huang, & Xu, 2012). DT-based machine-learners retain these useful statistical properties while improving predictive performance. An established single DT algorithm, C4.5 (Quinlan, 1993), ranks below LOG when benchmarked on balanced data sets (Brown & Mues, 2012). In contrast, the two leading DT-based machine-learners, Gradient Boosting Machine (GBM; Friedman, 2001) and RF (Breiman, 2001), demonstrated better predictive performance than either C4.5 or LOG (Brown & Mues, 2012).

Either GBM or RF could serve as the wrapper for BV selection. However, different algorithms exploit the same variables to different extents, and this is reflected in importance measures. GBM and RF consequently produce disparate variable importance assessments from identical data sets (Hastie, Tibshirani, & Friedman, 2009, p.593). To determine suitability these algorithms must be compared generally and in terms of variable importance.

Gradient Boosting Machine vs. Random Forest
-------------------------------------------

GBM and RF use 'meta-learning'. Multiple 'base-classifier' DTs are trained on a data set and later aggregated into a single 'meta-classifier' that makes the final predictions (Rokach, 2010, p.82). DT induction is so fast that meta-classifiers can include thousands of base-classifier trees. The algorithms use different approaches in creating training data sets for these many DTs. Each RF iteration 'bootstraps' (i.e. randomly sample with replacement) a data set equal in size to the original training data and 'subsamples' (i.e. randomly sample without replacement) a user-specified number of the available variables (Breiman, 2001). In contrast, each GBM iteration subsamples a fraction of the training data (e.g. 50% of records) and uses all variables (Friedman, 2002).

The most distinctive characteristic separating GBM from RF is 'gradient descent boosting'. 'Gradient descent' is an alternative means of parameterising a model. It reduces parameterisation to a scalar function that is iteratively optimised to decrease a chosen measure of error (Duda, Hart, & Stork, 2001, ch.1 p.16; ch.5 p.12). A 'gradient vector' of partial derivates is estimated from a differentiable function of a vector of variables. This function is the error value of a model with a set of arbitrary parameter starting values. Gradient descent iteratively shifts these values against the gradient direction until converging on a solution that minimises the error value. The magnitude of parameter adjustment per iteration is called 'step size' (Alpaydin, 2010, p.219) or 'shrinkage' (Ridgeway, 2012). Note that gradient 'ascent' iterates with the gradient direction to maximise a function (e.g. log likelihood). Mathematical theory refers to these iterative approaches as the 'method of conjugate gradients' (Zeidler, Hackbusch, & Schwarz, 2004, p.1063).

'Boosting' is a form of meta-learning developed by Schapire (1999) and is the method GBM uses to iteratively optimise. GBM trains a low accuracy base-classifier in the first iteration using equal weights for all records. In each subsequent iteration, it up-weights records misclassified in the previous iteration, progressively 'boosting' the representation of difficult to classify records. The base-classifiers from later iterations have a greater influence than those from earlier iterations when combined into the final meta-classifier (Williams, 2011, p.269-272; Hastie, Tibshirani, & Friedman, 2009, p.337-339).

The equivalent function used by RF, 'bootstrap aggregating' (i.e. 'bagging'), lacks this weighting. Bootstrapped records have equal weights, as do the resulting base-classifiers when aggregated into the meta-classifier (Breiman, 1996). Note that AdaBoost, the early boosting algorithm from which GBM originates, is effectively the same as RF if case weights are random rather than boosted (Breiman, 2001).

Four areas were reviewed to determine which algorithm, GBM or RF, was likely to provide the most reliable variable importance measures. Areas included prediction accuracy, literature presence, internal importance assessment method and assessment bias.

Prediction accuracy is one indication of how well an algorithm is exploiting the available variables. High accuracy implies optimal variable use. Results from GBM and RF comparisons are mixed and may be related to the software implementation used in testing. The 'Weka' (Hall et al., 2009) implementation of RF meets or outperforms the 'SAS Enterprise Miner' (SAS Institute Inc., 2012) equivalent to GBM, called 'gradient boosting' (Brown & Mues, 2012). Alternatively, GBM can match and outperform RF when using 'R' (R Core Team, 2013) implementations of both, 'gbm' and 'randomForest' respectively (Hastie et al., 2009, p.591). Each implementation performs differently. This experimental noise makes suitability judgements based on prediction accuracy premature, barring greater standardisation. Nevertheless, it is noteworthy that both GBM and RF exhibited high prediction accuracy compared to all other models tested in prior research.

In terms of literature presence, RF is common in 'feature selection' of genes, which is a bioinformatics equivalent of variable selection. Jiang et al. (2004) used RF to implement a 'gene shaving' protocol. Researchers divided the available genes (i.e. 'variables') into nested sets of increasing predictive value by repeatedly separating the least important 10% and remodelling those remaining. Diaz-Uriarte and de Andres (2006) similarly used RF to identify one set of highly predictive genes. "Given its performance and availability," write Diaz-Uriarte and de Andres, "variable selection using RF should probably become part of the standard tool-box of methods for the analysis of microarray data" (p.9). In contrast, GBM variable selection literature supports use in information technology applications, such as with search engine query data (Pan, Converse, Ahn, Salvetti, & Donato, 2009; Lu, Peng, Li, & Ahmed, 2006).

The machine-learning literature includes custom-programmed combinations of RF and GBM (Tuv, Borisov, Runger, & Torkkola, 2009). The justification cited by authors for including both algorithms is revealing. RF splits base-classifier DTs based on a random subsample of variables. Tuv et al. (2009) suggest that this makes RF less effective at identifying variable 'masking', where "one variable can effectively represent others in a model" (p.1342), thereby suppressing the importance score of that variable. This would indicate that GBM is marginally superior to RF for variable selection, although researchers from disparate fields continue to use both algorithms effectively.

GBM and RF employ different internal methods to assess variable importance. Each RF iteration finishes by using the recently trained DT to predict Out-Of-Bag (OOB) data set records. OOB records are those that, by chance, were not included in the bootstrapped sample for that DT. Before predicting, RF randomly permutes each variable, one at a time, in the OOB data set. The resulting percent increase in misclassification rate when each single variable is "noised up" is how RF assesses importance (Breiman, 2001, p.23-24). In contrast, GBM uses summed empirical improvement from splitting on a variable, averaged across all trees inducted (Ridgeway, 2012, p.5). The RF approach is more sophisticated, and because importance is calculated using OOB samples, some authors (Tuv et al., 2009, p.1347) suggest this is more accurate.

The final area of review is bias introduced to importance measures by each algorithm. RF can overestimate the splitting value of continuous variables and those with more categories (Strobl, Boulesteix, Zeileis, & Hothorn, 2007). This stems from the Gini Impurity splitting criteria, variable subsampling, and bootstrapping for base-classifier data sets, all of which are applied by the ubiquitous 'randomForest' R implementation (Liaw & Wiener, 2002). Bias is avoided by using a non-Gini splitting criteria, including all variables, and random sampling without replacement for base-classifier data sets. GBM, as least within the R implementation (Ridgeway & Southworth, 2013), already incorporates these elements. It uses an exponential loss function (i.e. twice binomial negative log-likelihood) ported from AdaBoost (Ridgeway, 2013). When set to a Bernoulli distribution, base-classifiers will split to minimise 'deviance' (i.e. of a new model from a saturated model) rather than Gini Impurity (Friedman, 2002). GBM also uses all possible variables and subsamples data sets without replacement. A bias not raised in the literature is the boosting used by GBM to up-weight misclassified records. This may inflate the importance of variables that predict difficult to classify records and requires further research.

To summarise, the research on accuracy and literature presence was mixed. RF benefits from more sophisticated internal importance measures, yet is prone to a unique bias destructive to those measures. This proves the deciding factor, indicating that GBM should produce more reliable importance measures. Both GBM and RF will perform well, if a single method must be chosen in research, GBM is the most suitable DT-based machine-learner for BV selection.

GBM Hyperparameters
-------------------

Gradient boosting can be applied via a Generalized Boosted Regression Model (aka. 'Gradient Boosting Machine'; GBM). This involves using the 'gbm' R package (Ridgeway & Southworth, 2013), which is a DT-based gradient descent boosting algorithm listed among R's machine-learning packages (Hothorn, 2013).

A well-fitting GBM will output the most accurate measures of variable importance. There are six customisable parameters that optimise model fit. These include loss function, sampling fraction, maximum interaction depth, step size, total number of trees and optimal iterations. Ridgeway (2012) is referenced extensively in determining suitable parameters.

The loss function must be set to a Bernoulli distribution, as appropriate for dichotomous classification, with predictions output as probabilities within a binary response variable range, 0 *GE* *p*(Default) *GE* 1.

The sampling fraction (i.e. 'bag fraction') is the proportion of the training set randomly sampled without replacement for base-classifier induction in each iteration. This can be left at the package default of 0.5 (i.e. 50% of the training data set).

Maximum interaction depth is typically set to '3'. This limits (i.e. preprunes) each base-classifier DT to three splits or 3-way interactions. When maximum interaction depth is left at the additive package default of '1', creating a 'decision stump' GBM, single variables dominate variable importance measures. The higher interaction depth allows better representation of variables that were not the most predictive, yet still made some contribution, and variables that contribute through complex interactions.

In each iteration, the gradient step for parameter estimation is multiplied by a fixed step size value, which is greater than zero and less than 1. This multiplication limits the magnitude of the update for each parameter adjustment. Smaller step size values reduce over-fitting and improve performance (Ridgeway, 2012). Alpaydin (2012, p.257) suggests a value of '0.02' or less, so step size can be left at the conservative package default of '0.001'.

The total number of trees required, synonymous with number of iterations required, is related to step size. A smaller step size value, meaning smaller gradient steps during parameter estimation, produces a meta-classifier that is more robust to over-fitting at the cost of requiring more trees to reach an optimal fit. Package simulations indicate that 10,000 trees are suitable for a step size value of '0.001' (Ridgeway, 2012).

Predictions can be made from any iteration of the GBM up to the final tree. Performance improves at each iteration or tree, reaches an optimal iteration with least error, and then progressively degrades from over-fitting. Accurate predictions require a GBM to predict using the most optimal iteration (i.e. best fitting tree). This optimal iteration can be determined via the *v*-fold cross validation feature, set to '5' folds, meaning that cross-validation error is estimated from five GBM models prior to a final GBM on all data. This feature can also be used to confirm whether the total number of trees was high enough to reach an optimal fit. Package functions can be used to extract the cross validation from the GBM R object and compute optimal iterations. Simulations indicate that this is the most accurate method, in contrast to Out-Of-Bag and Holdout Set estimates (Ridgeway, 2012).

GBM variable importance measures can be produced using two modelling procedures. The first included all BVs simultaneously in a large 'All-In-One' GBM. Variable importance measures are extracted at the completion of training. The second procedure, 'For-Loop', iterates through each BV to assess variable importance separately from other BVs. The loop repeatedly executed GBM training code and extracted variable importance measures until every BV is assessed in isolation from other BVs. The GBM models in both approaches should include the variables of *a priori* predictive value (e.g. age, sex).

All variables are available to GBM during training iterations and this makes the algorithm robust to variable masking (Tuv, Borisov, Runger, & Torkkola, 2009). Nevertheless, the 'For-Loop' can confirm that there are no large differences in the most important variables. The most predictive BVs selected by the 'All-In-One' GBM can be used within the final LOG model. This GBM more closely resembles the application scoring modelling environment (i.e. all variables used in one model).

References
----------

Alpaydin, E. (2012). Introduction to Machine Learning (2nd ed.). Massachusetts, USA: Massachusetts Institute of Technology.

Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32.

Breiman, L., Friedman, J.H., Olshen, R.A., Stone, C.J. (1984). Classification and Regression Trees. New York, USA: Chapman and Hall.

Brown, I., Mues, C. (2012). An experimental comparison of classification algorithms for imbalanced credit scoring data sets. Expert Systems with Applications, 39, 3446-3453.

Bucker, M., Kampen, M., Kramer, W. (2013). Reject inference in consumer credit scoring with nonignorable missing data. Journal of Banking & Finance, 37, 1040-1045.

de Ville, B. (2006). Decision Trees for Business Intelligence and Data Mining: Using SAS® Enterprise Miner™. North Carolina, USA: SAS Institute Inc.

Diaz-Uriarte, R., de Andres, S. (2006). Gene selection and classification of microarray data using random forest. BMC Bioinformatics, 7, 3.

Duda, R.O., Hart, P.E., Stork, D.G. (2001). Pattern classification (2nd ed.). New York, USA: Wiley

Dziuda, D.M. (2010). Data mining for genomics and proteomics: Analysis of gene and protein expression data. New Jersey, USA: John Wiley & Sons, Inc.

Engsted, T. (2009). Statistical vs. Economic Significance in Economics and Econometrics:

Friedman, J.H. (2001a). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29, 1189-1232.

Friedman, J.H. (2002). Stochastic gradient boosting. Computational Statistics & Data Analysis, 38, 367-378.

Guyon, I., Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157-1182.

Hall, M., Frank, E., Holmes, G., Pfahringer, B., Reutemann, P., Witten, I.H. (2009). The WEKA data mining software: An update, SIGKDD Explorations, 11, 10-18.

Hastie, T., Tibshirani, R., Friedman, J.H. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). New York: Springer.

Hothorn, T. (2013). CRAN Task View: Machine Learning & Statistical Learning. Retrieved December, 2013, from <http://cran.r-project.org/web/views/MachineLearning.html>

Jacobson, T., Roszbach, K. (2003). Bank lending policy, credit scoring and value-at-risk. Journal of Banking & Finance, 27, 615-633.

Jiang, H., Deng, Y., Chen, H.S., Tao, L., Sha, Q, Chen, J. Tsai1, C.J., Zhang, S. (2004). Joint analysis of two microarray gene-expression data sets to select lung adenocarcinoma marker genes. BMC Bioinformatics 2004, 5, 81.

Larose, D.T. (2005). Discovering knowledge in data: An introduction to data mining. Hoboken, New Jersey, USA: John Wiley & Sons, Inc.

Liaw, A., Wiener, M. (2002). Classification and Regression by randomForest. R News, 2, 18-22.

Lin, A.Z. (2013). Variable reduction in SAS by using weight of evidence and information value. SAS Global Forum 2013, Data Mining and Text Analytics. Retrieved December 2013, <http://support.sas.com/resources/papers/proceedings13/095-2013.pdf>

Lu, Y., Peng, F., Li, X., Ahmed, N. (2006). Coupling Feature Selection and Machine Learning Methods for Navigational Query Identification. Paper presented at Conference on Information and Knowledge Management, Arlington, Virginia, USA (pp.682-689).

Marin, J., Robert, C.P. (2014). Bayesian Essentials with R (2nd ed.). New York, USA: Springer.

Morgan, J.N., Sonquist, J.A. (1963). Problems in the Analysis of Survey Data, and a Proposal. Journal of the American Statistical Association, 58, 415-434.

Pan, F., Converse, T., Ahn, D., Salvetti, F., Donato, G. (2009). Feature Selection for Ranking using Boosted Trees. Paper presented at Conference on Information and Knowledge Management, Hong Kong, China.

Quinlan, J. R. (1993). C4.5 programs for machine learning. San Mateo, CA: Morgan Kaufmann.

R Core Team. (2013). R: A language and environment for statistical computing. R Foundation for Statistical Computing, Vienna, Austria.

Ridgeway, G. (2012). Generalized Boosted Models: A guide to the gbm package. Retrieved December 2013, <http://gradientboostedmodels.googlecode.com/git/gbm/inst/doc/gbm.pdf>

Ridgeway, G., Southworth, G. (2013). gbm: Generalized Boosted Regression Models (v2.1). Available from URL <http://CRAN.R-project.org/package=gbm>

Rokach, L. (2010). Pattern Classification Using Ensemble Methods. Singapore: World Scientific Publishing Co. Pte. Ltd.

Schapire, R.E. (1999), A brief introduction to boosting. Proceedings of the Sixteenth International Joint Conference on Artificial Intelligence, 1999.

Sharma, D. (2012). Improving the art, craft and science of economic credit risk scorecards using random forests: why credit scorers and economists should use random forests. Academy of Banking Studies Journal, 11, 93-115.

Sharma, D., Overstreet, G., Beling, P. (2009). Not if affordability data adds value but how to add real value by leveraging affordability data: Enhancing predictive capability of credit scoring using. Affordability Data. CAS (Casualty Actuarial Society) Working Paper.

Strobl, C., Boulesteix, A.L., Zeileis, A., Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. BMC Bioinformatics, 8, 25.

Thomas, L.C, Banasik, J, Crook, J.N. (2001). Recalibrating scorecards. Journal of Operational Research Society, 52, 981-968.

Tuv, E., Borisov, A., Runger, G., Torkkola, K. (2009). Feature Selection with Ensembles, Artificial Variables, and Redundancy Elimination. Journal of Machine Learning Research, 10, 1341-1366.

Wang, G., Ma, J., Huang, L., Xu, K. (2012). Two credit scoring models based on dual strategy ensemble trees. Knowledge-Based Systems, 26, 61-68.

Williams, G. (2011). Data mining with rattle and r: The art of excavating data for knowledge discovery. New York: Springer.

Zeidler, E., Hackbusch, W., Schwarz, H.R. (2004). Oxford Users' Guide to Mathematics. New York, USA: Oxford University Press Inc.
