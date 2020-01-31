import numpy as np
import shutil, os
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats

from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans, KMeans
from sklearn.mixture import GaussianMixture

class Evaluation:
    def __init__(self, settings, data):
        self.settings = settings
        self.data = data
        self.iterations = {}

    def read_iters(self):
        ## Load data from each saved iteration

        iters = int(np.ceil(self.settings.max_iter/self.settings.save_freq))
        inference_mu = np.zeros((self.settings.K,self.data.M,iters+2))
        inference_pi = np.zeros((self.data.N,self.settings.K,iters+2))
        inference_beta = np.zeros((self.settings.K,iters+2))
        inference_x = np.zeros((self.data.N,self.settings.K,self.data.M,iters+2))
        self.mu = []
        j = 0
        for i in sorted(os.listdir(self.settings.outdir)):
            if (i.find('model-') != -1) or i.find('init') == 0:
                toread = h5py.File(os.path.join(self.settings.outdir, i),'r')
                if "global_factor_features_mean" not in toread.keys():
                    continue
                mu_new =  toread["global_factor_features_mean"][:]
                print(inference_mu.shape, j, mu_new.shape)
                inference_mu[:,:,j] = mu_new
                inference_pi[:,:,j] = toread["local_factor_concentration"]
                inference_beta[:,j] = toread["global_factor_concentration"]
                inference_x[:,:,:,j] = toread["local_factor_features"]
                self.iterations[j] = i.split('-')[1].split('.')[0]
                print(i,j, self.iterations[j])
                j += 1
        self.inference_mu = inference_mu
        self.inference_pi = inference_pi
        self.inference_beta = inference_beta
        self.inference_x = inference_x

        # read loglikelihood
        self.loglikelihood = []
        for line in open(os.path.join(self.settings.outdir, "logfile.csv")):
            iteration, ELBO, loglikelihood = line.strip().split(',')
            if iteration != "iteration":
                self.loglikelihood.append(float(loglikelihood))

    def calculate_RMSE_mu(self, inference_mu):
        RMSE = np.zeros((self.settings.K, self.settings.K, len(self.iterations)))
        for i in range(len(self.iterations)):
            for k in range(0,self.settings.K):
                for kinf in range(0,self.settings.K):
                    RMSE[k,kinf,i] = np.sum(np.square(inference_mu[k,:,i] - self.data.mu[:][kinf,:]))
        return RMSE

    def plot_compare_mu(self,feature1,feature2):

        ## Run Other Models and plot results
        # Fuzzy K-Means
        mdl = FuzzyKMeans(k=self.settings.K)
        mdl.fit(self.data.obs[:])

        # K-Means
        kmeans = KMeans(k=self.settings.K)
        kmeans.fit(self.data.obs[:])

        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=self.settings.K)
        gmm.fit(self.data.obs[:])

        # Plot Results
        plt.clf()
        plt.plot(self.data.obs[:,feature1],self.data.obs[:,feature2],'ko',
                 markersize = 2, label = 'Data') # Observed Data
        plt.plot(self.inference_mu[:,feature1, len(self.iterations)-1],
                 self.inference_mu[:,feature2, len(self.iterations)-1],'b*',
                 markersize = 6, label = 'NDMs Inferred') # Inferred Centers
        plt.plot(self.data.mu[:,feature1],self.data.mu[:,feature2],'ro--',
                 markersize = 6, label = 'True') # True Centers
        plt.plot(self.data.mu[[-1,0],feature1],self.data.mu[[-1,0],feature2],
                 'r--') # Complete True Polygon
        plt.plot(mdl.cluster_centers_[:,feature1],mdl.cluster_centers_[:,feature2],'ks',
                 markersize = 6, label = 'Fuzzy K-Means')
        plt.plot(kmeans.cluster_centers_[:,feature1],kmeans.cluster_centers_[:,feature2],'kv',
                 markersize = 6, label = 'K-Means')
        plt.plot(gmm.means_[:,feature1],gmm.means_[:,feature2],'kh',markersize=4,label='GMM')

        plt.legend()
        plt.savefig(os.path.join(self.settings.outdir, 'Comparison_Graph.png'))

    def plot_mu_over_inf(self,feature1,feature2):
        plt.clf()
        plt.plot(self.data.mu[:,feature1],self.data.mu[:,feature2],'ro--',
                 markersize = 6, label = 'True') # True Centers
        plt.plot(self.data.mu[[-1,0],feature1],self.data.mu[[-1,0],feature2],
                 'r--') # Complete True Polygon
        plt.plot(self.data.obs[:,feature1],self.data.obs[:,feature2],'ko',
                 markersize = 2, label = 'Data') # Observed Data
        for k in range(0,self.settings.K):
            plt.plot(self.inference_mu[k,feature1,range(len(self.iterations))],
                     self.inference_mu[k,feature2,range(len(self.iterations))],
                     'o-',markersize=3,label = str(k))
        plt.plot(self.inference_mu[:,feature1,len(self.iterations)-1],
                 self.inference_mu[:,feature2,len(self.iterations)-1],'b*',
                 markersize = 6, label = 'NDMs Inferred') # Inferred Centers
        plt.legend()
        plt.savefig(os.path.join(self.settings.outdir, 'Mu_Inference.png'))

    def plot_x(self, feature1, feature2):
        x_dir = os.path.join(self.settings.outdir,'x')
        os.makedirs(x_dir)
        for i in sorted(self.iterations.keys()):
            plt.clf()
            fig = plt.figure(1)
            gridspec.GridSpec(6,5)
            plt.subplot2grid((6,5), (0,0), colspan=5, rowspan=5)
            #plt.subplot(311)
            for k in range(0,self.settings.K):
                plt.plot(self.data.x[:, k, feature1],
                         self.data.x[:, k, feature2],
                         'o',label = 'True ' + str(k), markersize = 2,
                         alpha = 0.5)
            for k in range(0,self.settings.K):
                plt.plot(self.inference_x[:, k, feature1, i],
                         self.inference_x[:, k, feature2, i],
                         'o',label = 'Inferred ' + str(k), markersize = 2,
                         alpha = 0.5)

            plt.plot(self.inference_mu[:, feature1, i],
                     self.inference_mu[:, feature2, i],'b*',
                     markersize = 6, label = 'NDMs Inferred') # Inferred Centers

            plt.plot(self.data.mu[:,feature1],self.data.mu[:, feature2],'ro--',
                     markersize = 6, label = 'True') # True Centers
            plt.plot(self.data.mu[[-1,0], feature1], self.data.mu[[-1,0], feature2],
                     'r--') # Complete True Polygon

            #plt.subplot(212)
            plt.subplot2grid((6,5), (5,0), colspan=5, rowspan=1)
            plt.plot(range(len(self.loglikelihood)), self.loglikelihood)
            if self.iterations[i] != 'final' and self.iterations[i] != 'init':
                plt.axvline(x=self.iterations[i])

            fig.tight_layout()
            fig.savefig(os.path.join(x_dir, 'x_inferences_%04d.png' % i))

    def plot_pi(self):
        pi_dir = os.path.join(self.settings.outdir,'pi')
        os.makedirs(pi_dir)
        plt.style.use('ggplot')
        width = 1.0
        samples = np.arange(10)
        for i in sorted(self.iterations.keys()):
            plt.clf()
            fig = plt.figure(1)
            gridspec.GridSpec(6,5)
            plt.subplot2grid((6,5), (0,0), colspan=5, rowspan=5)


            plt.barh(samples, self.inference_pi[samples,0,i], width)

            #plt.barh(samples, self.data.pi[samples,0], width, alpha = 1.0,fc=(1,1,1,0.5), edgecolor = 'black')

            pi_sum_data = self.data.pi[samples,0]
            pi_sum_fit = self.inference_pi[samples,0,i]
            for k in range(1, self.settings.K):
                plt.barh(samples, self.inference_pi[samples,k,i], width, left=pi_sum_fit)
                #plt.barh(samples, self.data.pi[samples,k], width,
                #         left=pi_sum_data, alpha = 1.0, fc=(1,1,1,0.5), edgecolor = 'black')

                pi_sum_fit += self.inference_pi[samples,k,i]

            plt.hold(True)
            plt.barh(samples, self.data.pi[samples,0], width, fc=(1,1,1,0.1), edgecolor = 'black')
            for k in range(1, self.settings.K):
                #plt.barh(samples, self.inference_pi[samples,k,i], width, left=pi_sum_fit)
                plt.barh(samples, self.data.pi[samples,k], width,
                         left=pi_sum_data, fc=(1,1,1,0.1), edgecolor = 'black')

                pi_sum_data += self.data.pi[samples,k]
                #pi_sum_fit += self.inference_pi[samples,k,i]
            plt.xlim((0.0,1.0))
            plt.ylim((0.0-width/2,len(samples)-width/2))

            plt.subplot2grid((6,5), (5,0), colspan=5, rowspan=1)
            plt.plot(range(len(self.loglikelihood)), self.loglikelihood)
            if self.iterations[i] != 'final':
                plt.axvline(x=self.iterations[i])

            fig.tight_layout()
            fig.savefig(os.path.join(pi_dir, 'pi_inferences_%04d.png' % i))

    def est_KL(self):
        mins = np.amin(self.data.obs,0) # col min
        maxs = np.amax(self.data.obs,0) # col max

        numPoints = 3 # number of points between min/max in grid

        ## Create grid of point values
        XXX = np.meshgrid(*[np.linspace(i,j,numPoints)[:-1] for i,j in zip(mins,maxs)])
        # As numpy
        dim = [self.data.M]
        for i in range(0,self.data.M):
            dim.append(numPoints-1)
        mesh = np.zeros(dim)
        for i in range(0,self.data.M):
            mesh[i] = XXX[i]
        mesh_list = np.reshape(mesh,(self.data.M,(numPoints-1)**self.data.M))
        #print(mesh_list.shape)
        #for i in range(0,(numPoints-1)**self.data.M):
            #print(mesh_list[:,i])

        # Probability under true parameters (based on pi and x)
        '''prob_mat = np.zeros((mesh_list.shape[1],self.data.N))
        print("Calculating True Probs")
        for j in range(0,prob_mat.shape[1]):
            #print("data point " + str(j))
            mean = np.matmul(self.data.pi[j,:],self.data.x[j,:,:])
            for i in range(0,prob_mat.shape[0]):
                prob_mat[i,j] = scipy.stats.multivariate_normal.pdf(mesh_list[:,i],mean = mean, cov = 1)
        #print(prob_mat.shape)
        prob_true = np.sum(prob_mat,axis=1)


        #print(prob_true.shape)
        #print(prob_true)

        thresh = .000001
        # Probability under learned parameters (based on pi and mu)
        mean = np.matmul(self.inference_pi[:,:,-1],self.inference_mu[:,:,-1])
        prob_mat = np.zeros((mesh_list.shape[1],self.data.N))
        print("Caculating Model Probs (Mu)")
        for j in range(0,prob_mat.shape[1]):
            #print("data point " + str(j))
            for i in range(0,prob_mat.shape[0]):
                prob_mat[i,j] = scipy.stats.multivariate_normal.pdf(mesh_list[:,i],mean = mean[j], cov = 1)
        prob_model = np.sum(prob_mat,axis=1)
        under_thresh = prob_model < thresh
        prob_model[under_thresh] = thresh
        #print(prob_model.shape)

        np.seterr(divide='ignore')
        p_over_q = np.log(np.divide(prob_true,prob_model))
        self.KL = np.sum(np.multiply(prob_true,p_over_q)) # How to deal with divide by zero
        print('KL from mu model')
        print(self.KL)

        # Probability under learned parameters (based on pi and x)
        prob_mat = np.zeros((mesh_list.shape[1],self.data.N))
        print("Caculating Model Probs (x)")
        for j in range(0,prob_mat.shape[1]):
            mean = np.matmul(self.inference_pi[j,:,-1],self.inference_x[j,:,:,-1])
            #print("data point " + str(j))
            for i in range(0,prob_mat.shape[0]):
                prob_mat[i,j] = scipy.stats.multivariate_normal.pdf(mesh_list[:,i],mean = mean, cov = 1)
        prob_model_x = np.sum(prob_mat,axis=1)
        under_thresh = prob_model_x < thresh
        prob_model_x[under_thresh] = thresh
        #print(prob_model_x.shape)

        np.seterr(divide='ignore')
        p_over_q_x = np.log(np.divide(prob_true,prob_model_x))
        self.KL_x = np.sum(np.multiply(prob_true,p_over_q_x)) # How to deal with divide by zero
        print('KL from x model')
        print(self.KL_x)'''

    def output(self, cleanup):
        self.read_iters()
        RMSE = self.calculate_RMSE_mu(self.inference_mu)
        self.plot_compare_mu(1,2)
        self.plot_mu_over_inf(1,2)
        self.plot_x(1,2)
        self.plot_pi()
        self.est_KL()
        print('True Mu')
        print(np.array(self.data.mu))
        print('Inferred Mu')
        print(self.inference_mu[:,:,len(self.iterations)-1])
        print('Mu RMSE ' + str(RMSE[:,:,len(self.iterations)-1]))
        #print(inference_beta)

        bashCommand = "convert -delay 10 %s %s" % \
                (os.path.join(self.settings.outdir, "x/x*.png"),
                os.path.join(self.settings.outdir, "x_inference.gif"))
        import subprocess
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        bashCommand = "convert -delay 10 %s %s" % \
                (os.path.join(self.settings.outdir, "pi/pi*.png"),
                os.path.join(self.settings.outdir, "pi_inference.gif"))
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

## To Do
# 1. Check if data.mu exists
# 3. How to match mus? -> Pi, Beta RMSEs
# 4. Output to File
# 5. Plot of x's
