/*
Copyright 2017 The Kubernetes Authors.

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

package set

import (
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	envutil "k8s.io/kubernetes/pkg/kubectl/cmd/util/env"
	"k8s.io/kubernetes/pkg/kubectl/resource"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

var (
	envResources = `
  	pod (po), replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs)`

	envLong = templates.LongDesc(`
		Update environment variables on a pod template.

		List environment variable definitions in one or more pods, pod templates.
		Add, update, or remove container environment variable definitions in one or
		more pod templates (within replication controllers or deployment configurations).
		View or modify the environment variable definitions on all containers in the
		specified pods or pod templates, or just those that match a wildcard.

		If "--env -" is passed, environment variables can be read from STDIN using the standard env
		syntax.

		Possible resources include (case insensitive):
		` + envResources)

	envExample = templates.Examples(`
		# Update deployment 'registry' with a new environment variable
	  kubectl set env deployment/registry STORAGE_DIR=/local

	  # List the environment variables defined on a deployments 'sample-build'
	  kubectl set env deployment/sample-build --list

	  # List the environment variables defined on all pods
	  kubectl set env pods --all --list

	  # Output modified deployment in YAML, and does not alter the object on the server
	  kubectl set env deployment/sample-build STORAGE_DIR=/data -o yaml

	  # Update all containers in all replication controllers in the project to have ENV=prod
	  kubectl set env rc --all ENV=prod

	  # Import environment from a secret
	  kubectl set env --from=secret/mysecret deployment/myapp

	  # Import environment from a config map with a prefix
	  kubectl set env --from=configmap/myconfigmap --prefix=MYSQL_ deployment/myapp

	  # Remove the environment variable ENV from container 'c1' in all deployment configs
	  kubectl set env deployments --all --containers="c1" ENV-

	  # Remove the environment variable ENV from a deployment definition on disk and
	  # update the deployment config on the server
	  kubectl set env -f deploy.json ENV-

	  # Set some of the local shell environment into a deployment config on the server
	  env | grep RAILS_ | kubectl set env -e - deployment/registry`)
)

type EnvOptions struct {
	Out io.Writer
	Err io.Writer
	In  io.Reader

	resource.FilenameOptions
	EnvParams []string
	EnvArgs   []string
	Resources []string

	All         bool
	Resolve     bool
	List        bool
	ShortOutput bool
	Local       bool
	Overwrite   bool
	DryRun      bool

	ResourceVersion   string
	ContainerSelector string
	Selector          string
	Output            string
	From              string
	Prefix            string

	Mapper  meta.RESTMapper
	Typer   runtime.ObjectTyper
	Builder *resource.Builder
	Infos   []*resource.Info
	Encoder runtime.Encoder

	Cmd *cobra.Command

	UpdatePodSpecForObject func(obj runtime.Object, fn func(*api.PodSpec) error) (bool, error)
	PrintObject            func(cmd *cobra.Command, isLocal bool, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error
}

// NewCmdEnv implements the OpenShift cli env command
func NewCmdEnv(f cmdutil.Factory, in io.Reader, out, errout io.Writer) *cobra.Command {
	options := &EnvOptions{
		Out: out,
		Err: errout,
		In:  in,
	}
	cmd := &cobra.Command{
		Use:     "env RESOURCE/NAME KEY_1=VAL_1 ... KEY_N=VAL_N",
		Short:   "Update environment variables on a pod template",
		Long:    envLong,
		Example: fmt.Sprintf(envExample),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Complete(f, cmd, args))
			cmdutil.CheckErr(options.RunEnv(f))
		},
	}
	usage := "the resource to update the env"
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmd.Flags().StringVarP(&options.ContainerSelector, "containers", "c", "*", "The names of containers in the selected pod templates to change - may use wildcards")
	cmd.Flags().StringP("from", "", "", "The name of a resource from which to inject environment variables")
	cmd.Flags().StringP("prefix", "", "", "Prefix to append to variable names")
	cmd.Flags().StringArrayVarP(&options.EnvParams, "env", "e", options.EnvParams, "Specify a key-value pair for an environment variable to set into each container.")
	cmd.Flags().BoolVar(&options.List, "list", options.List, "If true, display the environment and any changes in the standard format. this flag will removed when we have kubectl view env.")
	cmd.Flags().BoolVar(&options.Resolve, "resolve", options.Resolve, "If true, show secret or configmap references when listing variables")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", options.Selector, "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set env will NOT contact api-server but run locally.")
	cmd.Flags().BoolVar(&options.All, "all", options.All, "If true, select all resources in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&options.Overwrite, "overwrite", true, "If true, allow environment to be overwritten, otherwise reject updates that overwrite existing environment.")

	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddPrinterFlags(cmd)

	return cmd
}

func validateNoOverwrites(existing []api.EnvVar, env []api.EnvVar) error {
	for _, e := range env {
		if current, exists := findEnv(existing, e.Name); exists && current.Value != e.Value {
			return fmt.Errorf("'%s' already has a value (%s), and --overwrite is false", current.Name, current.Value)
		}
	}
	return nil
}

func keyToEnvName(key string) string {
	validEnvNameRegexp := regexp.MustCompile("[^a-zA-Z0-9_]")
	return strings.ToUpper(validEnvNameRegexp.ReplaceAllString(key, "_"))
}

func (o *EnvOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	resources, envArgs, ok := envutil.SplitEnvironmentFromResources(args)
	if !ok {
		return cmdutil.UsageErrorf(o.Cmd, "all resources must be specified before environment changes: %s", strings.Join(args, " "))
	}
	if len(o.Filenames) == 0 && len(resources) < 1 {
		return cmdutil.UsageErrorf(cmd, "one or more resources must be specified as <resource> <name> or <resource>/<name>")
	}

	o.Mapper, o.Typer = f.Object()
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.Encoder = f.JSONEncoder()
	o.ContainerSelector = cmdutil.GetFlagString(cmd, "containers")
	o.List = cmdutil.GetFlagBool(cmd, "list")
	o.Resolve = cmdutil.GetFlagBool(cmd, "resolve")
	o.Selector = cmdutil.GetFlagString(cmd, "selector")
	o.All = cmdutil.GetFlagBool(cmd, "all")
	o.Overwrite = cmdutil.GetFlagBool(cmd, "overwrite")
	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.From = cmdutil.GetFlagString(cmd, "from")
	o.Prefix = cmdutil.GetFlagString(cmd, "prefix")
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.PrintObject = f.PrintObject

	o.EnvArgs = envArgs
	o.Resources = resources
	o.Cmd = cmd

	o.ShortOutput = cmdutil.GetFlagString(cmd, "output") == "name"

	if o.List && len(o.Output) > 0 {
		return cmdutil.UsageErrorf(o.Cmd, "--list and --output may not be specified together")
	}

	return nil
}

// RunEnv contains all the necessary functionality for the OpenShift cli env command
func (o *EnvOptions) RunEnv(f cmdutil.Factory) error {
	kubeClient, err := f.ClientSet()
	if err != nil {
		return err
	}

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	env, remove, err := envutil.ParseEnv(append(o.EnvParams, o.EnvArgs...), o.In)
	if err != nil {
		return err
	}

	if len(o.From) != 0 {
		b := f.NewBuilder().
			ContinueOnError().
			NamespaceParam(cmdNamespace).DefaultNamespace().
			FilenameParam(enforceNamespace, &o.FilenameOptions).
			Flatten()

		if !o.Local {
			b = b.
				SelectorParam(o.Selector).
				ResourceTypeOrNameArgs(o.All, o.From).
				Latest()
		} else {
			b = b.Local(f.ClientForMapping)
		}

		infos, err := b.Do().Infos()
		if err != nil {
			return err
		}

		for _, info := range infos {
			switch from := info.Object.(type) {
			case *api.Secret:
				for key := range from.Data {
					envVar := api.EnvVar{
						Name: keyToEnvName(key),
						ValueFrom: &api.EnvVarSource{
							SecretKeyRef: &api.SecretKeySelector{
								LocalObjectReference: api.LocalObjectReference{
									Name: from.Name,
								},
								Key: key,
							},
						},
					}
					env = append(env, envVar)
				}
			case *api.ConfigMap:
				for key := range from.Data {
					envVar := api.EnvVar{
						Name: keyToEnvName(key),
						ValueFrom: &api.EnvVarSource{
							ConfigMapKeyRef: &api.ConfigMapKeySelector{
								LocalObjectReference: api.LocalObjectReference{
									Name: from.Name,
								},
								Key: key,
							},
						},
					}
					env = append(env, envVar)
				}
			default:
				return fmt.Errorf("unsupported resource specified in --from")
			}
		}
	}

	if len(o.Prefix) != 0 {
		for i := range env {
			env[i].Name = fmt.Sprintf("%s%s", o.Prefix, env[i].Name)
		}
	}

	b := f.NewBuilder().
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		Flatten()

	if !o.Local {
		b = b.
			SelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, o.Resources...).
			Latest()
	} else {
		b = b.Local(f.ClientForMapping)
	}

	o.Infos, err = b.Do().Infos()
	if err != nil {
		return err
	}
	patches := CalculatePatches(o.Infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *api.PodSpec) error {
			resolutionErrorsEncountered := false
			containers, _ := selectContainers(spec.Containers, o.ContainerSelector)
			if len(containers) == 0 {
				fmt.Fprintf(o.Err, "warning: %s/%s does not have any containers matching %q\n", info.Mapping.Resource, info.Name, o.ContainerSelector)
				return nil
			}
			for _, c := range containers {
				if !o.Overwrite {
					if err := validateNoOverwrites(c.Env, env); err != nil {
						return err
					}
				}

				c.Env = updateEnv(c.Env, env, remove)
				if o.List {
					resolveErrors := map[string][]string{}
					store := envutil.NewResourceStore()

					fmt.Fprintf(o.Out, "# %s %s, container %s\n", info.Mapping.Resource, info.Name, c.Name)
					for _, env := range c.Env {
						// Print the simple value
						if env.ValueFrom == nil {
							fmt.Fprintf(o.Out, "%s=%s\n", env.Name, env.Value)
							continue
						}

						// Print the reference version
						if !o.Resolve {
							fmt.Fprintf(o.Out, "# %s from %s\n", env.Name, envutil.GetEnvVarRefString(env.ValueFrom))
							continue
						}

						value, err := envutil.GetEnvVarRefValue(kubeClient, cmdNamespace, store, env.ValueFrom, info.Object, c)
						// Print the resolved value
						if err == nil {
							fmt.Fprintf(o.Out, "%s=%s\n", env.Name, value)
							continue
						}

						// Print the reference version and save the resolve error
						fmt.Fprintf(o.Out, "# %s from %s\n", env.Name, envutil.GetEnvVarRefString(env.ValueFrom))
						errString := err.Error()
						resolveErrors[errString] = append(resolveErrors[errString], env.Name)
						resolutionErrorsEncountered = true
					}

					// Print any resolution errors
					errs := []string{}
					for err, vars := range resolveErrors {
						sort.Strings(vars)
						errs = append(errs, fmt.Sprintf("error retrieving reference for %s: %v", strings.Join(vars, ", "), err))
					}
					sort.Strings(errs)
					for _, err := range errs {
						fmt.Fprintln(o.Err, err)
					}
				}
			}
			if resolutionErrorsEncountered {
				return errors.New("failed to retrieve valueFrom references")
			}
			return nil
		})

		if err == nil {
			// TODO: switch UpdatePodSpecForObject to work on v1.PodSpec, use info.VersionedObject, and avoid conversion completely
			versionedEncoder := legacyscheme.Codecs.EncoderForVersion(o.Encoder, info.Mapping.GroupVersionKind.GroupVersion())
			return runtime.Encode(versionedEncoder, info.Object)
		}
		return nil, err
	})

	if o.List {
		return nil
	}

	allErrs := []error{}

	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			allErrs = append(allErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}

		// no changes
		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			continue
		}

		if o.PrintObject != nil && (o.Local || o.DryRun) {
			if err := o.PrintObject(o.Cmd, o.Local, o.Mapper, info.Object, o.Out); err != nil {
				return err
			}
			continue
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch env update to pod template: %v\n", err))
			continue
		}
		info.Refresh(obj, true)

		// make sure arguments to set or replace environment variables are set
		// before returning a successful message
		if len(env) == 0 && len(o.EnvArgs) == 0 {
			return fmt.Errorf("at least one environment variable must be provided")
		}

		if len(o.Output) > 0 {
			if err := o.PrintObject(o.Cmd, o.Local, o.Mapper, obj, o.Out); err != nil {
				return err
			}
			continue
		}

		cmdutil.PrintSuccess(o.Mapper, o.ShortOutput, o.Out, info.Mapping.Resource, info.Name, false, "env updated")
	}
	return utilerrors.NewAggregate(allErrs)
}
