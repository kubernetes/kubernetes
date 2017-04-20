package cmd

import (
	"fmt"
	"github.com/blang/semver"
	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"
	"io"
	"io/ioutil"
	"k8s.io/client-go/pkg/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	"net/http"
	"os"
	"regexp"
)

func NewConfigurator(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiext.MasterConfiguration{}
	api.Scheme.Default(cfg)
	etcdDiscveryService := ""
	outFile := ""
	cmd := &cobra.Command{
		Use:   "configure",
		Short: "Run this in order to generate masters configurations",
		Run: func(cmd *cobra.Command, args []string) {
			api.Scheme.Default(cfg)
			internalcfg := kubeadmapi.MasterConfiguration{}
			api.Scheme.Convert(cfg, &internalcfg, nil)
			err := setDefaultConfiguration(&internalcfg)
			kubeadmutil.CheckErr(err)
			internalcfg.MasterCertificates = &kubeadmapi.MasterCertificates{}
			err = certs.GeneratePKIAssets(&internalcfg)
			kubeadmutil.CheckErr(err)
			if etcdDiscveryService != "" {
				if internalcfg.Count < 1 {
					internalcfg.Count = 1
				}
				token, err := getDiscoveryToken(etcdDiscveryService, internalcfg.Count)
				kubeadmutil.CheckErr(err)
				internalcfg.Etcd.Discovery = token
			}
			confData, err := yaml.Marshal(&internalcfg)
			kubeadmutil.CheckErr(err)
			if outFile == "" {
				out.Write(confData)
				fmt.Fprintf(out, "\n")
			} else {
				fileOut, err := os.Create(outFile)
				kubeadmutil.CheckErr(err)
				fileOut.Write(confData)
				fileOut.Close()
				fmt.Fprintf(out, "Result saved to %s\n", outFile)

			}
		},
	}
	cmd.PersistentFlags().StringVar(
		&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress,
		"The IP address the API Server will advertise it's listening on. 0.0.0.0 means the default network interface's address.",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.PublicAddress, "public-address", cfg.PublicAddress,
		"The PublicAddress address e.g. LoadBalancer",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.HostnameOverride, "hostname-override", cfg.HostnameOverride,
		"Override node name.",
	)
	cmd.PersistentFlags().IntVar(
		&cfg.Count, "master-count", 1,
		"Master count",
	)
	cmd.PersistentFlags().Int32Var(
		&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort,
		"Port for the API Server to bind to",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet,
		"Use alternative range of IP address for service VIPs",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet,
		"Specify range of IP addresses for the pod network; if set, the control plane will automatically allocate CIDRs for every node",
	)
	cmd.PersistentFlags().StringVar(
		&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain,
		`Use alternative domain for services, e.g. "myorg.internal"`,
	)
	cmd.PersistentFlags().StringVar(
		&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane`,
	)
	cmd.PersistentFlags().StringVar(
		&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir,
		`The path where to save and store the certificates`,
	)
	cmd.PersistentFlags().StringSliceVar(
		&cfg.APIServerCertSANs, "apiserver-cert-extra-sans", cfg.APIServerCertSANs,
		`Optional extra altnames to use for the API Server serving cert. Can be both IP addresses and dns names.`,
	)
	cmd.PersistentFlags().StringVar(
		&etcdDiscveryService, "etcd-discovery", "https://discovery.etcd.io",
		`Use etcd discovery service to join etcd nodes to cluster.`,
	)
	cmd.PersistentFlags().StringVar(
		&outFile, "out", "",
		`May be save configureation to this file?`,
	)
	cmd.PersistentFlags().StringVar(
		&cfg.ClusterName, "cluster-name", cfg.ClusterName,
		"Cluster name. Used for tagging cloud provider resources")
	return cmd
}
func getDiscoveryToken(url string, size int) (string, error) {
	resp, err := http.Get(fmt.Sprintf("%s/new?size=%v", url, size))

	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Failed allocate discovery token from %s reason - % ", url, resp.Status)
	}
	token, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	var urlRegex = regexp.MustCompile(`^(https?://)?([\da-z\.-]+)\.([a-z\.]{2,6})([/\w \.-]*)*/?$`)

	if urlRegex.Match(token) {
		return string(token), nil
	}
	return "", fmt.Errorf("Failed get discovery token, got %q", token)
}

func setDefaultConfiguration(cfg *kubeadmapi.MasterConfiguration) error {
	ver, err := kubeadmutil.KubernetesReleaseVersion(cfg.KubernetesVersion)
	if err != nil {
		return err
	}
	cfg.KubernetesVersion = ver
	// Omit the "v" in the beginning, otherwise semver will fail
	k8sVersion, err := semver.Parse(cfg.KubernetesVersion[1:])
	if err != nil {
		return fmt.Errorf("couldn't parse kubernetes version %q: %v", cfg.KubernetesVersion, err)
	}
	if k8sVersion.LT(minK8sVersion) {
		return fmt.Errorf("this version of kubeadm only supports deploying clusters with the control plane version >= v%s. Current version: %s", kubeadmconstants.MinimumControlPlaneVersion, cfg.KubernetesVersion)
	}

	if cfg.Token == "" {
		var err error
		cfg.Token, err = tokenutil.GenerateToken()
		if err != nil {
			return fmt.Errorf("couldn't generate random token: %v", err)
		}
	}

	return nil
}
