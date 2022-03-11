**Humboldt University of Berlin - Department of Informatics and Mathematics..**

****

# Mean-field theory of the predictive coding spiking neural network
MSc thesis under the supervision of [Professor Tilo Schwalger](http://page.math.tu-berlin.de/~schwalge/).
The full text is available in the repository under predictive_coding_thesis.pdf.

## Introduction
Biological neurons are challenged with the enormous task to reliably process large amounts of information in a highly uncertain environment. Indeed, neuronal activity - encoded through sequences of action potentials (or spikes) - is notorious for its ubiquitous, Poisson-like spiking variability. A central problem in computational neuroscience is to explain how efficient coding can be performed under such conditions.
	
A naive approach could be to obtain reliable outcomes through averages over large populations of equivalent neurons. This could minimize the effects of biophysical noise on individual neurons. However, the overall effect of such microscopic fluctuations is too weak to account for the full extent of cortical variability. Moreover, one would reasonably hope that evolution has come up with more sophisticated ways to perform calculations than simple averaging over a large amount of redundant units. 
	
This has led researchers to consider the global network activity itself as a possible source of fluctuations. Using mean-field theory, where stochastic variables that depend on population-level activity are replaced by their average values, the dynamics of large networks can be derived analytically. Such an approximation becomes exact in the large N-limit, where the number of neurons tends to infinity, and can yield important insights for realistic population sizes as well. In particular, it has been shown that the dynamics of a large network of leaky integrate-and-firing (LIF) neurons stabilize globally, while individual firing rates are still highly irregular.
	
While these theoretical results successfully reconciled stable, reliable dynamics with the strong heterogeneity observed in cortical recordings, the precise mechanisms and reasons which would lead to such a regime remained unclear. More recently, a powerful framework has been introduced by [Boerlin et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003258), who demonstrate that efficient coding and balanced dynamics are signatures of networks that follow the paradigm of predictive coding. In particular, the neural network generates real-time predictions of some dynamic sensory input, capable of implementing arbitrary linear dynamical systems. From the biological point of view, the resulting prediction errors can be propagated to other cortical areas and have been shown to play an important role in various contexts such as audiovisual speech, motor control or learning of causal relationships in general.
	
The predictive coding model gives a compelling account of neural computations. Biologically plausible leaky integrate-and-fire dynamics arise automatically from the optimization of a given objective function. This functional approach ensures that spikes are used optimally to minimize the prediction error. Moreover, neurons exhibit high Poisson-like variability, while still ensuring high coding accuracy with supperclassical error scaling, with the error decreasing as 1/N with the number of neurons. 

In turn, these results have spurred an impressive amount of consecutive research. Very recently, first theoretical results have also been established by [Kadmon et al.](https://arxiv.org/abs/2006.14178) for the model using mean-field theory. However, the approach is based entirely on firing rates, with artificially imposed constraints to fit the predictive coding framework. Therefore, results cannot be applied directly to the original framework, nor realistic spiking models (as they are observed in the brain) in general. 
	
In this work, we bridge the gap between theoretical rate models and realistic spiking networks by deriving mean-field results for a Linear-Nonlinear Poisson approximation of the original model developed by Boerlin et al.

