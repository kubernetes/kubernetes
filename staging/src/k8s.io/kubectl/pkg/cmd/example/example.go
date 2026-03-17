/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package example

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	yaml "sigs.k8s.io/yaml"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/explain"
	utilcomp "k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	longDesc = templates.LongDesc(i18n.T(`
        Output a practical example manifest for a resource.

        Resolves the provided TYPE using discovery (like 'kubectl explain') and prints
        a sensible, immediately applicable YAML manifest you can pipe to 'kubectl apply -f -'.`))

	examples = templates.Examples(i18n.T(`
        # Example Pod manifest
        kubectl example pod

        # List supported example types
        kubectl example --list

        # Parameterize name/replicas where applicable
        kubectl example deployment --replicas=3 --name=web`))
)

var buildersByKind = map[string]func(string, string, int) ([]byte, error){
	"pod": func(name, image string, _ int) ([]byte, error) {
		return yaml.Marshal(buildPod(name, image))
	},
	"pods": func(name, image string, _ int) ([]byte, error) {
		return yaml.Marshal(buildPod(name, image))
	},
	"po": func(name, image string, _ int) ([]byte, error) {
		return yaml.Marshal(buildPod(name, image))
	},
	"deployment": func(name, image string, replicas int) ([]byte, error) {
		return yaml.Marshal(buildDeployment(name, image, replicas))
	},
	"deployments": func(name, image string, replicas int) ([]byte, error) {
		return yaml.Marshal(buildDeployment(name, image, replicas))
	},
	"deploy": func(name, image string, replicas int) ([]byte, error) {
		return yaml.Marshal(buildDeployment(name, image, replicas))
	},
	"service": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildService(name))
	},
	"services": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildService(name))
	},
	"svc": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildService(name))
	},
	"persistentvolumeclaim": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildPVC(name))
	},
	"persistentvolumeclaims": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildPVC(name))
	},
	"pvc": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildPVC(name))
	},
	"secret": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildSecret(name))
	},
	"secrets": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildSecret(name))
	},
	"customresourcedefinition": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildCRD(name))
	},
	"customresourcedefinitions": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildCRD(name))
	},
	"crd": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildCRD(name))
	},
	"configmap": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildConfigMap(name))
	},
	"configmaps": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildConfigMap(name))
	},
	"cm": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildConfigMap(name))
	},
	"job": func(name, image string, _ int) ([]byte, error) {
		return yaml.Marshal(buildJob(name, image))
	},
	"jobs": func(name, image string, _ int) ([]byte, error) {
		return yaml.Marshal(buildJob(name, image))
	},
	"cronjob": func(name, image string, _ int) ([]byte, error) {
		return yaml.Marshal(buildCronJob(name, image))
	},
	"cronjobs": func(name, image string, _ int) ([]byte, error) {
		return yaml.Marshal(buildCronJob(name, image))
	},
	"ingress": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildIngress(name))
	},
	"ingresses": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildIngress(name))
	},
	"ing": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildIngress(name))
	},
	"networkpolicy": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildNetworkPolicy(name))
	},
	"networkpolicies": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildNetworkPolicy(name))
	},
	"netpol": func(name, _ string, _ int) ([]byte, error) {
		return yaml.Marshal(buildNetworkPolicy(name))
	},
}

// Flags represent CLI flags for the command.
type Flags struct {
	List     bool
	Name     string
	Image    string
	Replicas int

	genericiooptions.IOStreams
}

// NewFlags creates default flags.
func NewFlags(streams genericiooptions.IOStreams) *Flags {
	return &Flags{
		IOStreams: streams,
	}
}

// AddFlags binds flags to the cobra command.
func (f *Flags) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVar(&f.List, "list", f.List, "List supported resource examples")
	cmd.Flags().StringVar(&f.Name, "name", f.Name, "Override metadata.name in the example where applicable")
	cmd.Flags().StringVar(&f.Image, "image", f.Image, "Override container image in the example where applicable")
	cmd.Flags().IntVar(&f.Replicas, "replicas", f.Replicas, "Override replicas where applicable (deployments, etc.)")
}

// Options represent runtime options for the command.
type Options struct {
	Flags

	Factory cmdutil.Factory
	Mapper  meta.RESTMapper

	rawArgs []string
}

// ToOptions converts CLI flags to runtime options.
func (f *Flags) ToOptions(factory cmdutil.Factory, args []string) (*Options, error) {
	return &Options{
		Flags:   *f,
		Factory: factory,
		Mapper:  nil,
		rawArgs: args,
	}, nil
}

// Validate validates inputs.
func (o *Options) Validate() error {
	if o.List {
		return nil
	}
	if len(o.rawArgs) == 0 {
		return fmt.Errorf("you must specify the type of resource to example. %s", cmdutil.SuggestAPIResources("kubectl"))
	}
	if len(o.rawArgs) > 1 {
		return fmt.Errorf("we accept only this format: example TYPE")
	}
	return nil
}

// Run executes the command.
func (o *Options) Run() error {
	if o.List {
		return printSupported(o.Out)
	}

	userToken := o.rawArgs[0]
	// First, attempt static offline fallback without any discovery.
	if _, key, ok := fallbackResolve(userToken); ok {
		builder := buildersByKind[key]
		rendered, err := builder(defaultNameFor(key, o.Name), o.Image, o.Replicas)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(o.Out, strings.TrimSpace(string(rendered)))
		return err
	}

	// Otherwise, resolve via RESTMapper and discovery like 'kubectl explain'.
	// If no usable kubeconfig is present, avoid discovery to prevent noisy errors.
	if !hasUsableConfig(o.Factory) {
		return fmt.Errorf("no example available for %q. Try --list or 'kubectl explain %s'", userToken, userToken)
	}
	if o.Mapper == nil {
		var err error
		o.Mapper, err = o.Factory.ToRESTMapper()
		if err != nil {
			return err
		}
	}
	fullySpecifiedGVR, _, err := explain.SplitAndParseResourceRequestWithMatchingPrefix(userToken, o.Mapper)
	var key string
	if err == nil {
		gvk, kindErr := o.Mapper.KindFor(fullySpecifiedGVR)
		if kindErr != nil || gvk.Empty() {
			gvk, kindErr = o.Mapper.KindFor(fullySpecifiedGVR.GroupResource().WithVersion(""))
		}
		if kindErr == nil && !gvk.Empty() {
			kindKey := strings.ToLower(gvk.Kind)
			if _, ok := buildersByKind[kindKey]; ok {
				key = kindKey
			}
		}
		if key == "" {
			resKey := fullySpecifiedGVR.Resource
			if _, ok := buildersByKind[resKey]; ok {
				key = resKey
			}
		}
	}

	if key == "" {
		return fmt.Errorf("no example available for %q. Try --list or 'kubectl explain %s'", userToken, userToken)
	}

	builder := buildersByKind[key]
	rendered, err := builder(defaultNameFor(key, o.Name), o.Image, o.Replicas)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintln(o.Out, strings.TrimSpace(string(rendered)))
	return err
}

func defaultNameFor(kindLower, override string) string {
	if override != "" {
		return override
	}
	base := kindLower
	if strings.HasSuffix(base, "s") {
		base = strings.TrimSuffix(base, "s")
	}
	return "example-" + base
}

func printSupported(out io.Writer) error {
	var keys []string
	seen := map[string]struct{}{}
	for k := range buildersByKind {
		// Only print singular canonical kinds once
		switch k {
		case "pods", "deployments", "services", "persistentvolumeclaims", "secrets", "customresourcedefinitions",
			"configmaps", "jobs", "cronjobs", "ingresses", "networkpolicies":
			// skip plurals in listing to avoid duplicates
			continue
		}
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, err := fmt.Fprintln(out, k); err != nil {
			return err
		}
	}
	return nil
}

// NewCmdExample returns the cobra command for 'kubectl example'.
func NewCmdExample(parent string, f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	flags := NewFlags(streams)
	cmd := &cobra.Command{
		Use:                   "example TYPE [--list] [--name=...] [--image=...] [--replicas=N]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Get a practical example manifest for a resource"),
		Long:                  longDesc + "\n\n" + cmdutil.SuggestAPIResources(parent),
		Example:               examples,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(f, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}
	flags.AddFlags(cmd)
	cmd.ValidArgsFunction = utilcomp.ResourceTypeAndNameCompletionFunc(f)
	return cmd
}

// hasUsableConfig returns true if the factory can provide a kubeconfig with a
// non-empty current context and server URL. It does not contact the server.
func hasUsableConfig(f cmdutil.Factory) bool {
	loader := f.ToRawKubeConfigLoader()
	cfg, err := loader.RawConfig()
	if err != nil {
		return false
	}
	if cfg.CurrentContext == "" {
		return false
	}
	ctx, ok := cfg.Contexts[cfg.CurrentContext]
	if !ok || ctx == nil {
		return false
	}
	cluster, ok := cfg.Clusters[ctx.Cluster]
	if !ok || cluster == nil {
		return false
	}
	return cluster.Server != ""
}

// fallbackResolve attempts to resolve common resource tokens offline without discovery.
// Returns a synthetic GroupVersionKind and the buildersByKind key if successful.
func fallbackResolve(token string) (schema.GroupVersionKind, string, bool) {
	t := strings.ToLower(strings.TrimSpace(token))
	switch t {
	case "po", "pod", "pods":
		return schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}, "pod", true
	case "deploy", "deployment", "deployments":
		return schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, "deployment", true
	case "svc", "service", "services":
		return schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, "service", true
	case "pvc", "persistentvolumeclaim", "persistentvolumeclaims":
		return schema.GroupVersionKind{Group: "", Version: "v1", Kind: "PersistentVolumeClaim"}, "persistentvolumeclaim", true
	case "secret", "secrets":
		return schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Secret"}, "secret", true
	case "crd", "customresourcedefinition", "customresourcedefinitions":
		return schema.GroupVersionKind{Group: "apiextensions.k8s.io", Version: "v1", Kind: "CustomResourceDefinition"}, "customresourcedefinition", true
	case "cm", "configmap", "configmaps":
		return schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"}, "configmap", true
	case "job", "jobs":
		return schema.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"}, "job", true
	case "cronjob", "cronjobs":
		return schema.GroupVersionKind{Group: "batch", Version: "v1", Kind: "CronJob"}, "cronjob", true
	case "ing", "ingress", "ingresses":
		return schema.GroupVersionKind{Group: "networking.k8s.io", Version: "v1", Kind: "Ingress"}, "ingress", true
	case "netpol", "networkpolicy", "networkpolicies":
		return schema.GroupVersionKind{Group: "networking.k8s.io", Version: "v1", Kind: "NetworkPolicy"}, "networkpolicy", true
	default:
		return schema.GroupVersionKind{}, "", false
	}
}
