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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/cobra"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/storage/names"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

const (
	volumePrefix = "volume-"
)

var (
	volumeLong = templates.LongDesc(`
		Update volumes on a pod template

		This command can add, update or remove volumes from containers for any object
		that has a pod template (deployment configs, replication controllers, or pods).
		You can list volumes in pod or any object that has a pod template. You can
		specify a single object or multiple, and alter volumes on all containers or
		just those that match a given name.

		If you alter a volume setting on a deployment, a deployment will be
		triggered. Changing a replication controller will not affect running pods, and
		you cannot change a pod's volumes once it has been created.

		Volume types include:

		emptydir (empty directory) *default*: A directory allocated when the pod is
		created on a local host, is removed when the pod is deleted and is not copied
		across servers

		hostdir (host directory): A directory with specific path on any host
		(requires elevated privileges)

		persistentvolumeclaim or pvc (persistent volume claim): Link the volume
		directory in the container to a persistent volume claim you have allocated by
		name - a persistent volume claim is a request to allocate storage. Note that
		if your claim hasn't been bound, your pods will not start.

		secret (mounted secret): Secret volumes mount a named secret to the provided
		directory.

		For descriptions on other volume types, see https://kubernetes.io/`)

	volumeExample = templates.Examples(`
	  # List volumes defined on all deployment in the current namespace
	  kubectl set volume deployment --all

	  # Add a new empty dir volume to deployment 'registry' mounted under
	  # /var/lib/registry
	  kubectl set volume deployment/registry --add --mount-path=/var/lib/registry

	  # Use an existing persistent volume claim (pvc) to overwrite an existing volume 'v1'
	  kubectl set volume deployment/registry --add --name=v1 -t pvc --claim-name=pvc1 --overwrite

	  # Remove volume 'v1' from deployment 'registry'
	  kubectl set volume deployment/registry --remove --name=v1

	  # Create a new persistent volume claim that overwrites an existing volume 'v1'
	  kubectl set volume deployment/registry --add --name=v1 -t pvc --claim-size=1G --overwrite

	  # Change the mount point for volume 'v1' to /data
	  kubectl set volume deployment/registry --add --name=v1 -m /data --overwrite

	  # Modify the deployment by removing volume mount "v1" from container "c1"
	  # (and by removing the volume "v1" if no other containers have volume mounts that reference it)
	  kubectl set volume deployment/registry --remove --name=v1 --containers=c1

	  # Add new volume based on a more complex volume source (Git repo, AWS EBS, GCE PD,
	  # Ceph, Gluster, NFS, ISCSI, ...)
	  kubectl set volume deployment/registry --add -m /repo --source=<json-string>`)
)

type VolumeOptions struct {
	DefaultNamespace       string
	EnforceNamespace       bool
	Out                    io.Writer
	Err                    io.Writer
	Mapper                 meta.RESTMapper
	RESTClientFactory      func(mapping *meta.RESTMapping) (resource.RESTClient, error)
	UpdatePodSpecForObject func(obj runtime.Object, fn func(*v1.PodSpec) error) (bool, error)
	Client                 coreclient.PersistentVolumeClaimsGetter
	Encoder                runtime.Encoder
	Cmd                    *cobra.Command

	// Resource selection
	resource.FilenameOptions
	Selector string
	All      bool

	// Operations
	Add    bool
	Remove bool
	List   bool

	// Common optional params
	Name        string
	Containers  string
	Confirm     bool
	Local       bool
	DryRun      bool
	Output      string
	PrintObject func(cmd *cobra.Command, isLocal bool, mapper meta.RESTMapper, obj runtime.Object, out io.Writer) error

	// Add op params
	AddOpts *AddVolumeOptions
}

type AddVolumeOptions struct {
	Type          string
	MountPath     string
	SubPath       string
	DefaultMode   string
	Overwrite     bool
	Path          string
	ConfigMapName string
	SecretName    string
	Source        string

	CreateClaim bool
	ClaimName   string
	ClaimSize   string
	ClaimMode   string
	ClaimClass  string

	TypeChanged bool
}

func NewCmdVolume(f cmdutil.Factory, out, errOut io.Writer) *cobra.Command {
	addOpts := &AddVolumeOptions{}
	opts := &VolumeOptions{AddOpts: addOpts}
	cmd := &cobra.Command{
		Use:     "volumes RESOURCE/NAME --add",
		Short:   "Update volumes on a pod template",
		Long:    volumeLong,
		Example: volumeExample,
		Aliases: []string{"volume"},
		Run: func(cmd *cobra.Command, args []string) {
			addOpts.TypeChanged = cmd.Flag("type").Changed

			cmdutil.CheckErr(opts.Validate(cmd, args))
			cmdutil.CheckErr(opts.Complete(f, cmd, out, errOut))
			cmdutil.CheckErr(opts.RunVolume(args, f))
		},
	}
	usage := "the resource to update the volume"
	cmdutil.AddFilenameOptionFlags(cmd, &opts.FilenameOptions, usage)
	cmd.Flags().StringVarP(&opts.Selector, "selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&opts.All, "all", false, "If true, select all resources in the namespace of the specified resource types")
	cmd.Flags().BoolVar(&opts.Add, "add", false, "If true, add volume and/or volume mounts for containers")
	cmd.Flags().BoolVar(&opts.Remove, "remove", false, "If true, remove volume and/or volume mounts for containers")
	cmd.Flags().BoolVar(&opts.List, "list", false, "If true, list volumes and volume mounts for containers")
	cmd.Flags().BoolVar(&opts.Local, "local", false, "If true, set volume will NOT contact api-server but run locally.")

	cmd.Flags().StringVar(&opts.Name, "name", "", "Name of the volume. If empty, auto generated for add operation")
	cmd.Flags().StringVarP(&opts.Containers, "containers", "c", "*", "The names of containers in the selected pod templates to change - may use wildcards")
	cmd.Flags().BoolVar(&opts.Confirm, "confirm", false, "If true, confirm that you really want to remove multiple volumes")

	cmd.Flags().StringVarP(&addOpts.Type, "type", "t", "", "Type of the volume source for add operation. Supported options: emptyDir, hostPath, secret, configmap, persistentVolumeClaim")
	cmd.Flags().StringVarP(&addOpts.MountPath, "mount-path", "m", "", "Mount path inside the container. Optional param for --add or --remove")
	cmd.Flags().StringVar(&addOpts.SubPath, "sub-path", "", "Path within the local volume from which the container's volume should be mounted. Optional param for --add or --remove")
	cmd.Flags().StringVarP(&addOpts.DefaultMode, "default-mode", "", "", "The default mode bits to create files with. Can be between 0000 and 0777. Defaults to 0644.")
	cmd.Flags().BoolVar(&addOpts.Overwrite, "overwrite", false, "If true, replace existing volume source with the provided name and/or volume mount for the given resource")
	cmd.Flags().StringVar(&addOpts.Path, "path", "", "Host path. Must be provided for hostPath volume type")
	cmd.Flags().StringVar(&addOpts.ConfigMapName, "configmap-name", "", "Name of the persisted config map. Must be provided for configmap volume type")
	cmd.Flags().StringVar(&addOpts.SecretName, "secret-name", "", "Name of the persisted secret. Must be provided for secret volume type")
	cmd.Flags().StringVar(&addOpts.ClaimName, "claim-name", "", "Persistent volume claim name. Must be provided for persistentVolumeClaim volume type")
	cmd.Flags().StringVar(&addOpts.ClaimClass, "claim-class", "", "StorageClass to use for the persistent volume claim")
	cmd.Flags().StringVar(&addOpts.ClaimSize, "claim-size", "", "If specified along with a persistent volume type, create a new claim with the given size in bytes. Accepts SI notation: 10, 10G, 10Gi")
	cmd.Flags().StringVar(&addOpts.ClaimMode, "claim-mode", "ReadWriteOnce", "Set the access mode of the claim to be created. Valid values are ReadWriteOnce (rwo), ReadWriteMany (rwm), or ReadOnlyMany (rom)")
	cmd.Flags().StringVar(&addOpts.Source, "source", "", "Details of volume source as json string. This can be used if the required volume type is not supported by --type option. (e.g.: '{\"gitRepo\": {\"repository\": <git-url>, \"revision\": <commit-hash>}}')")

	cmdutil.AddDryRunFlag(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmd.MarkFlagFilename("filename", "yaml", "yml", "json")

	// deprecate --list option
	cmd.Flags().MarkDeprecated("list", "Volumes and volume mounts can be listed by providing a resource with no additional options.")

	return cmd
}

func (o *VolumeOptions) Validate(cmd *cobra.Command, args []string) error {
	if len(o.Selector) > 0 {
		if _, err := labels.Parse(o.Selector); err != nil {
			return errors.New("--selector=<selector> must be a valid label selector")
		}
		if o.All {
			return errors.New("you may specify either --selector or --all but not both")
		}
	}
	if len(o.FilenameOptions.Filenames) == 0 && len(args) < 1 {
		return errors.New("provide one or more resources to add, list, or delete volumes on as TYPE/NAME")
	}

	numOps := 0
	if o.Add {
		numOps++
	}
	if o.Remove {
		numOps++
	}
	if o.List {
		numOps++
	}

	switch {
	case numOps == 0:
		o.List = true
	case numOps > 1:
		return errors.New("you may only specify one operation at a time")
	}

	output := cmdutil.GetFlagString(cmd, "output")
	if o.List && len(output) > 0 {
		return errors.New("--list and --output may not be specified together")
	}

	err := o.AddOpts.Validate(o.Add)
	if err != nil {
		return err
	}
	// Removing all volumes for the resource type needs confirmation
	if o.Remove && len(o.Name) == 0 && !o.Confirm {
		return errors.New("must provide --confirm for removing more than one volume")
	}
	return nil
}

func (a *AddVolumeOptions) Validate(isAddOp bool) error {
	if isAddOp {
		if len(a.Type) == 0 && (len(a.ClaimName) > 0 || len(a.ClaimSize) > 0) {
			a.Type = "persistentvolumeclaim"
			a.TypeChanged = true
		}
		if len(a.Type) == 0 && (len(a.SecretName) > 0) {
			a.Type = "secret"
			a.TypeChanged = true
		}
		if len(a.Type) == 0 && (len(a.ConfigMapName) > 0) {
			a.Type = "configmap"
			a.TypeChanged = true
		}
		if len(a.Type) == 0 && (len(a.Path) > 0) {
			a.Type = "hostpath"
			a.TypeChanged = true
		}
		if len(a.Type) == 0 {
			a.Type = "emptydir"
		}

		if len(a.Type) == 0 && len(a.Source) == 0 {
			return errors.New("must provide --type or --source for --add operation")
		} else if a.TypeChanged && len(a.Source) > 0 {
			return errors.New("either specify --type or --source but not both for --add operation")
		}

		if len(a.Type) > 0 {
			return validateObjectType(a)
		} else if len(a.Path) > 0 || len(a.SecretName) > 0 || len(a.ClaimName) > 0 {
			return errors.New("--path|--secret-name|--claim-name are only valid for --type option")
		}

		if len(a.Source) > 0 {
			var source map[string]interface{}
			err := json.Unmarshal([]byte(a.Source), &source)
			if err != nil {
				return err
			}
			if len(source) > 1 {
				return errors.New("must provide only one volume for --source")
			}

			var vs v1.VolumeSource
			err = json.Unmarshal([]byte(a.Source), &vs)
			if err != nil {
				return err
			}
		}
		if len(a.ClaimClass) > 0 {
			selectedLowerType := strings.ToLower(a.Type)
			if selectedLowerType != "persistentvolumeclaim" && selectedLowerType != "pvc" {
				return errors.New("must provide --type as persistentVolumeClaim")
			}
			if len(a.ClaimSize) == 0 {
				return errors.New("must provide --claim-size to create new pvc with claim-class")
			}
		}
	} else if len(a.Source) > 0 || len(a.Path) > 0 || len(a.SecretName) > 0 || len(a.ConfigMapName) > 0 || len(a.ClaimName) > 0 || a.Overwrite {
		return errors.New("--type|--path|--configmap-name|--secret-name|--claim-name|--source|--overwrite are only valid for --add operation")
	}
	return nil
}

func (o *VolumeOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, out, errOut io.Writer) error {
	client, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	if err != nil {
		return err
	}
	o.Client = client.Core()

	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	o.Output = cmdutil.GetFlagString(cmd, "output")
	o.PrintObject = f.PrintObject

	o.Cmd = cmd
	o.DefaultNamespace = cmdNamespace
	o.EnforceNamespace = enforceNamespace
	o.Out = out
	o.Err = errOut
	o.Mapper, _ = f.Object()
	o.RESTClientFactory = f.ClientForMapping
	o.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	o.Encoder = f.JSONEncoder()
	o.DryRun = cmdutil.GetDryRunFlag(o.Cmd)

	// In case of volume source ignore the default volume type
	if len(o.AddOpts.Source) > 0 {
		o.AddOpts.Type = ""
	}
	if len(o.AddOpts.ClaimSize) > 0 {
		o.AddOpts.CreateClaim = true
		if len(o.AddOpts.ClaimName) == 0 {
			o.AddOpts.ClaimName = names.SimpleNameGenerator.GenerateName("pvc-")
		}
		q, err := apiresource.ParseQuantity(o.AddOpts.ClaimSize)
		if err != nil {
			return fmt.Errorf("--claim-size is not valid: %v", err)
		}
		o.AddOpts.ClaimSize = q.String()
	}
	if len(o.AddOpts.DefaultMode) == 0 {
		o.AddOpts.DefaultMode = "644"
	}
	switch strings.ToLower(o.AddOpts.ClaimMode) {
	case strings.ToLower(string(v1.ReadOnlyMany)), "rom":
		o.AddOpts.ClaimMode = string(v1.ReadOnlyMany)
	case strings.ToLower(string(v1.ReadWriteOnce)), "rwo":
		o.AddOpts.ClaimMode = string(v1.ReadWriteOnce)
	case strings.ToLower(string(v1.ReadWriteMany)), "rwm":
		o.AddOpts.ClaimMode = string(v1.ReadWriteMany)
	case "":
	default:
		return errors.New("--claim-mode must be one of ReadWriteOnce (rwo), ReadWriteMany (rwm), or ReadOnlyMany (rom)")
	}
	return nil
}

func (o *VolumeOptions) RunVolume(args []string, f cmdutil.Factory) error {
	builder := f.NewBuilder().
		ContinueOnError().
		NamespaceParam(o.DefaultNamespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.FilenameOptions).
		Flatten()

	if !o.Local {
		builder = builder.
			LabelSelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	} else {
		// if a --local flag was provided, and a resource was specified in the form
		// <resource>/<name>, fail immediately as --local cannot query the api server
		// for the specified resource.
		if len(args) > 0 {
			return resource.LocalResourceError
		}

		builder = builder.Local(f.ClientForMapping)
	}

	singleItemImplied := false
	infos, err := builder.Do().IntoSingleItemImplied(&singleItemImplied).Infos()
	if err != nil {
		return err
	}

	if o.List {
		listingErrors := o.printVolumes(infos)
		if len(listingErrors) > 0 {
			return cmdutil.ErrExit
		}
		return nil
	}

	updateInfos := []*resource.Info{}
	// if a claim should be created, generate the info we'll add to the flow
	if o.Add && o.AddOpts.CreateClaim {
		claim := o.AddOpts.createClaim()
		m, err := o.Mapper.RESTMapping(schema.GroupKind{Kind: "PersistentVolumeClaim"})
		if err != nil {
			return err
		}
		mapper := resource.ClientMapperFunc(o.RESTClientFactory)
		client, err := mapper.ClientForMapping(m)
		if err != nil {
			return err
		}
		info := &resource.Info{
			Mapping:   m,
			Client:    client,
			Namespace: o.DefaultNamespace,
			Object:    claim,
		}
		infos = append(infos, info)
		updateInfos = append(updateInfos, info)
	}

	patches, patchError := o.getVolumeUpdatePatches(infos, singleItemImplied)

	if patchError != nil {
		return patchError
	}
	if len(o.Output) > 0 || o.Local || o.DryRun {
		allErrs := []error{}
		for _, info := range infos {
			if err := o.PrintObject(o.Cmd, o.Local, o.Mapper, info.VersionedObject, o.Out); err != nil {
				allErrs = append(allErrs, fmt.Errorf("print Object error:%v\n", err))
			}
		}
		if len(allErrs) == 0 {
			return nil
		} else {
			return utilerrors.NewAggregate(allErrs)
		}
	}

	failed := false
	allErrs := []error{}

	for _, info := range updateInfos {
		var obj runtime.Object
		if len(info.ResourceVersion) == 0 {
			obj, err = resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, false, info.Object)
		} else {
			obj, err = resource.NewHelper(info.Client, info.Mapping).Replace(info.Namespace, info.Name, true, info.Object)
		}
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch volume update to pod template: %v\n", err))
			failed = true
			continue
		}
		info.Refresh(obj, true)
		fmt.Fprintf(o.Out, "%s/%s\n", info.Mapping.Resource, info.Name)
	}
	for _, patch := range patches {
		info := patch.Info
		if patch.Err != nil {
			failed = true
			allErrs = append(allErrs, fmt.Errorf("error: %s/%s %v\n", info.Mapping.Resource, info.Name, patch.Err))
			continue
		}

		if string(patch.Patch) == "{}" || len(patch.Patch) == 0 {
			allErrs = append(allErrs, fmt.Errorf("info: %s %q was not changed\n", info.Mapping.Resource, info.Name))
			failed = true
			continue
		}

		glog.V(4).Infof("Calculated patch %s", patch.Patch)

		obj, err := resource.NewHelper(info.Client, info.Mapping).Patch(info.Namespace, info.Name, types.StrategicMergePatchType, patch.Patch)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("failed to patch volume update to pod template: %v\n", err))
			failed = true
			continue
		}

		info.Refresh(obj, true)
		cmdutil.PrintSuccess(o.Mapper, false, o.Out, info.Mapping.Resource, info.Name, false, "updated")
	}
	if failed {
		return utilerrors.NewAggregate(allErrs)
	}
	return nil
}

func (o *VolumeOptions) getVolumeUpdatePatches(infos []*resource.Info, singleItemImplied bool) ([]*Patch, error) {
	skipped := 0
	patches := CalculatePatches(infos, o.Encoder, func(info *resource.Info) ([]byte, error) {
		transformed := false
		ok, err := o.UpdatePodSpecForObject(info.VersionedObject, func(spec *v1.PodSpec) error {
			var e error
			switch {
			case o.Add:
				e = o.addVolumeToSpec(spec, info, singleItemImplied)
				transformed = true
			case o.Remove:
				e = o.removeVolumeFromSpec(spec, info)
				transformed = true
			}
			return e
		})
		if !ok {
			skipped++
		}
		if err == nil {
			return runtime.Encode(o.Encoder, info.VersionedObject)
		}
		return nil, err
	})
	if singleItemImplied && skipped == len(infos) {
		patchError := fmt.Errorf("the %s %s is not a pod or does not have a pod template", infos[0].Mapping.Resource, infos[0].Name)
		return patches, patchError
	}
	return patches, nil
}

func setVolumeSourceByType(kv *v1.Volume, opts *AddVolumeOptions) error {
	switch strings.ToLower(opts.Type) {
	case "emptydir":
		kv.EmptyDir = &v1.EmptyDirVolumeSource{}
	case "hostpath":
		kv.HostPath = &v1.HostPathVolumeSource{
			Path: opts.Path,
		}
	case "secret":
		defaultMode, err := strconv.ParseUint(opts.DefaultMode, 8, 32)
		if err != nil {
			return err
		}
		defaultMode32 := int32(defaultMode)
		kv.Secret = &v1.SecretVolumeSource{
			SecretName:  opts.SecretName,
			DefaultMode: &defaultMode32,
		}
	case "configmap":
		defaultMode, err := strconv.ParseUint(opts.DefaultMode, 8, 32)
		if err != nil {
			return err
		}
		defaultMode32 := int32(defaultMode)
		kv.ConfigMap = &v1.ConfigMapVolumeSource{
			LocalObjectReference: v1.LocalObjectReference{
				Name: opts.ConfigMapName,
			},
			DefaultMode: &defaultMode32,
		}
	case "persistentvolumeclaim", "pvc":
		kv.PersistentVolumeClaim = &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: opts.ClaimName,
		}
	default:
		return fmt.Errorf("invalid volume type: %s", opts.Type)
	}
	return nil
}

func (o *VolumeOptions) printVolumes(infos []*resource.Info) []error {
	listingErrors := []error{}
	for _, info := range infos {
		_, err := o.UpdatePodSpecForObject(info.Object, func(spec *v1.PodSpec) error {
			return o.listVolumeForSpec(spec, info)
		})
		if err != nil {
			listingErrors = append(listingErrors, err)
			fmt.Fprintf(o.Err, "error: %s/%s %v\n", info.Mapping.Resource, info.Name, err)
		}
	}
	return listingErrors
}

func (v *AddVolumeOptions) createClaim() *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: v.ClaimName,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.PersistentVolumeAccessMode(v.ClaimMode)},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): apiresource.MustParse(v.ClaimSize),
				},
			},
		},
	}
	if len(v.ClaimClass) > 0 {
		pvc.Annotations = map[string]string{
			v1.BetaStorageClassAnnotation: v.ClaimClass,
		}
	}
	return pvc
}

func (o *VolumeOptions) setVolumeSource(kv *v1.Volume) error {
	var err error
	opts := o.AddOpts
	if len(opts.Type) > 0 {
		err = setVolumeSourceByType(kv, opts)
	} else if len(opts.Source) > 0 {
		err = json.Unmarshal([]byte(opts.Source), &kv.VolumeSource)
	}
	return err
}

func (o *VolumeOptions) setVolumeMount(spec *v1.PodSpec, info *resource.Info) error {
	opts := o.AddOpts
	containers, _ := selectContainers(spec.Containers, o.Containers)
	if len(containers) == 0 && o.Containers != "*" {
		fmt.Fprintf(o.Err, "warning: %s/%s does not have any containers matching %q\n", info.Mapping.Resource, info.Name, o.Containers)
		return nil
	}

	for _, c := range containers {
		for _, m := range c.VolumeMounts {
			if path.Clean(m.MountPath) == path.Clean(opts.MountPath) && m.Name != o.Name {
				return fmt.Errorf("volume mount '%s' already exists for container '%s'", opts.MountPath, c.Name)
			}
		}
		for i, m := range c.VolumeMounts {
			if m.Name == o.Name {
				c.VolumeMounts = append(c.VolumeMounts[:i], c.VolumeMounts[i+1:]...)
				break
			}
		}
		volumeMount := &v1.VolumeMount{
			Name:      o.Name,
			MountPath: path.Clean(opts.MountPath),
		}
		if len(opts.SubPath) > 0 {
			volumeMount.SubPath = path.Clean(opts.SubPath)
		}
		c.VolumeMounts = append(c.VolumeMounts, *volumeMount)
	}
	return nil
}

func (o *VolumeOptions) getVolumeName(spec *v1.PodSpec, singleResource bool) (string, error) {
	opts := o.AddOpts
	if opts.Overwrite {
		// Multiple resources can have same mount-path for different volumes,
		// so restrict it for single resource to uniquely find the volume
		if !singleResource {
			return "", fmt.Errorf("you must specify --name for the volume name when dealing with multiple resources")
		}
		if len(opts.MountPath) > 0 {
			containers, _ := selectContainers(spec.Containers, o.Containers)
			var name string
			matchCount := 0
			for _, c := range containers {
				for _, m := range c.VolumeMounts {
					if path.Clean(m.MountPath) == path.Clean(opts.MountPath) {
						name = m.Name
						matchCount += 1
						break
					}
				}
			}

			switch matchCount {
			case 0:
				return "", fmt.Errorf("unable to find the volume for mount-path: %s", opts.MountPath)
			case 1:
				return name, nil
			default:
				return "", fmt.Errorf("found multiple volumes with same mount-path: %s", opts.MountPath)
			}
		} else {
			return "", fmt.Errorf("ambiguous --overwrite, specify --name or --mount-path")
		}
	} else { // Generate volume name
		name := names.SimpleNameGenerator.GenerateName(volumePrefix)
		if len(o.Output) == 0 {
			fmt.Fprintf(o.Err, "info: Generated volume name: %s\n", name)
		}
		return name, nil
	}
}

func (o *VolumeOptions) addVolumeToSpec(spec *v1.PodSpec, info *resource.Info, singleResource bool) error {
	opts := o.AddOpts
	if len(o.Name) == 0 {
		var err error
		if o.Name, err = o.getVolumeName(spec, singleResource); err != nil {
			return err
		}
	}
	newVolume := &v1.Volume{
		Name: o.Name,
	}
	setSource := true
	vNameFound := false
	for i, vol := range spec.Volumes {
		if o.Name == vol.Name {
			vNameFound = true
			if !opts.Overwrite {
				return fmt.Errorf("volume '%s' already exists. Use --overwrite to replace", o.Name)
			}
			if !opts.TypeChanged && len(opts.Source) == 0 {
				newVolume.VolumeSource = vol.VolumeSource
				setSource = false
			}
			spec.Volumes = append(spec.Volumes[:i], spec.Volumes[i+1:]...)
			break
		}
	}

	// if --overwrite was passed, but volume did not previously
	// exist, log a warning that no volumes were overwritten
	if !vNameFound && opts.Overwrite && len(o.Output) == 0 {
		fmt.Fprintf(o.Err, "warning: volume %q did not previously exist and was not overriden. A new volume with this name has been created instead.", o.Name)
	}

	if setSource {
		err := o.setVolumeSource(newVolume)
		if err != nil {
			return err
		}
	}
	spec.Volumes = append(spec.Volumes, *newVolume)

	if len(opts.MountPath) > 0 {
		err := o.setVolumeMount(spec, info)
		if err != nil {
			return err
		}
	}
	return nil
}

func (o *VolumeOptions) removeSpecificVolume(spec *v1.PodSpec, containers, skippedContainers []*v1.Container) error {
	for _, c := range containers {
		for i, m := range c.VolumeMounts {
			if o.Name == m.Name {
				c.VolumeMounts = append(c.VolumeMounts[:i], c.VolumeMounts[i+1:]...)
				break
			}
		}
	}

	// Remove volume if no container is using it
	found := false
	for _, c := range skippedContainers {
		for _, m := range c.VolumeMounts {
			if o.Name == m.Name {
				found = true
				break
			}
		}
		if found {
			break
		}
	}
	if !found {
		foundVolume := false
		for i, vol := range spec.Volumes {
			if o.Name == vol.Name {
				spec.Volumes = append(spec.Volumes[:i], spec.Volumes[i+1:]...)
				foundVolume = true
				break
			}
		}
		if !foundVolume {
			return fmt.Errorf("volume '%s' not found", o.Name)
		}
	}
	return nil
}

func (o *VolumeOptions) removeVolumeFromSpec(spec *v1.PodSpec, info *resource.Info) error {
	containers, skippedContainers := selectContainers(spec.Containers, o.Containers)
	if len(containers) == 0 && o.Containers != "*" {
		fmt.Fprintf(o.Err, "warning: %s/%s does not have any containers matching %q\n", info.Mapping.Resource, info.Name, o.Containers)
		return nil
	}

	if len(o.Name) == 0 {
		for _, c := range containers {
			c.VolumeMounts = []v1.VolumeMount{}
		}
		spec.Volumes = []v1.Volume{}
	} else {
		err := o.removeSpecificVolume(spec, containers, skippedContainers)
		if err != nil {
			return err
		}
	}
	return nil
}

func sourceAccessMode(readOnly bool) string {
	if readOnly {
		return " read-only"
	}
	return ""
}

func describePersistentVolumeClaim(claim *v1.PersistentVolumeClaim) string {
	if len(claim.Spec.VolumeName) == 0 {
		// TODO: check for other dimensions of request - IOPs, etc
		if val, ok := claim.Spec.Resources.Requests[v1.ResourceStorage]; ok {
			return fmt.Sprintf("waiting for %sB allocation", val.String())
		}
		return "waiting to allocate"
	}
	// TODO: check for other dimensions of capacity?
	if val, ok := claim.Status.Capacity[v1.ResourceStorage]; ok {
		return fmt.Sprintf("allocated %sB", val.String())
	}
	return "allocated unknown size"
}

func describeVolumeSource(source *v1.VolumeSource) string {
	switch {
	case source.AWSElasticBlockStore != nil:
		return fmt.Sprintf("AWS EBS %s type=%s partition=%d%s", source.AWSElasticBlockStore.VolumeID, source.AWSElasticBlockStore.FSType, source.AWSElasticBlockStore.Partition, sourceAccessMode(source.AWSElasticBlockStore.ReadOnly))
	case source.EmptyDir != nil:
		return "empty directory"
	case source.GCEPersistentDisk != nil:
		return fmt.Sprintf("GCE PD %s type=%s partition=%d%s", source.GCEPersistentDisk.PDName, source.GCEPersistentDisk.FSType, source.GCEPersistentDisk.Partition, sourceAccessMode(source.GCEPersistentDisk.ReadOnly))
	case source.GitRepo != nil:
		if len(source.GitRepo.Revision) == 0 {
			return fmt.Sprintf("Git repository %s", source.GitRepo.Repository)
		}
		return fmt.Sprintf("Git repository %s @ %s", source.GitRepo.Repository, source.GitRepo.Revision)
	case source.Glusterfs != nil:
		return fmt.Sprintf("GlusterFS %s:%s%s", source.Glusterfs.EndpointsName, source.Glusterfs.Path, sourceAccessMode(source.Glusterfs.ReadOnly))
	case source.HostPath != nil:
		return fmt.Sprintf("host path %s", source.HostPath.Path)
	case source.ISCSI != nil:
		return fmt.Sprintf("ISCSI %s target-portal=%s type=%s lun=%d%s", source.ISCSI.IQN, source.ISCSI.TargetPortal, source.ISCSI.FSType, source.ISCSI.Lun, sourceAccessMode(source.ISCSI.ReadOnly))
	case source.NFS != nil:
		return fmt.Sprintf("NFS %s:%s%s", source.NFS.Server, source.NFS.Path, sourceAccessMode(source.NFS.ReadOnly))
	case source.PersistentVolumeClaim != nil:
		return fmt.Sprintf("pvc/%s%s", source.PersistentVolumeClaim.ClaimName, sourceAccessMode(source.PersistentVolumeClaim.ReadOnly))
	case source.RBD != nil:
		return fmt.Sprintf("Ceph RBD %v type=%s image=%s pool=%s%s", source.RBD.CephMonitors, source.RBD.FSType, source.RBD.RBDImage, source.RBD.RBDPool, sourceAccessMode(source.RBD.ReadOnly))
	case source.Secret != nil:
		return fmt.Sprintf("secret/%s", source.Secret.SecretName)
	default:
		return "unknown"
	}
}

func (o *VolumeOptions) listVolumeForSpec(spec *v1.PodSpec, info *resource.Info) error {
	containers, _ := selectContainers(spec.Containers, o.Containers)
	if len(containers) == 0 && o.Containers != "*" {
		fmt.Fprintf(o.Err, "warning: %s/%s does not have any containers matching %q\n", info.Mapping.Resource, info.Name, o.Containers)
		return nil
	}

	fmt.Fprintf(o.Out, "%s/%s\n", info.Mapping.Resource, info.Name)
	checkName := (len(o.Name) > 0)
	found := false
	for _, vol := range spec.Volumes {
		if checkName && o.Name != vol.Name {
			continue
		}
		found = true

		refInfo := ""
		if vol.VolumeSource.PersistentVolumeClaim != nil {
			claimName := vol.VolumeSource.PersistentVolumeClaim.ClaimName
			claim, err := o.Client.PersistentVolumeClaims(info.Namespace).Get(claimName, metav1.GetOptions{})
			switch {
			case err == nil:
				refInfo = fmt.Sprintf("(%s)", describePersistentVolumeClaim(claim))
			case apierrs.IsNotFound(err):
				refInfo = "(does not exist)"
			default:
				fmt.Fprintf(o.Err, "error: unable to retrieve persistent volume claim %s referenced in %s/%s: %v", claimName, info.Mapping.Resource, info.Name, err)
			}
		}
		if len(refInfo) > 0 {
			refInfo = " " + refInfo
		}

		fmt.Fprintf(o.Out, "  %s%s as %s\n", describeVolumeSource(&vol.VolumeSource), refInfo, vol.Name)
		for _, c := range containers {
			for _, m := range c.VolumeMounts {
				if vol.Name != m.Name {
					continue
				}
				if len(spec.Containers) == 1 {
					fmt.Fprintf(o.Out, "    mounted at %s\n", m.MountPath)
				} else {
					fmt.Fprintf(o.Out, "    mounted at %s in container %s\n", m.MountPath, c.Name)
				}
			}
		}
	}
	if checkName && !found {
		return fmt.Errorf("volume %q not found", o.Name)
	}

	return nil
}

func validateObjectType(a *AddVolumeOptions) error {
	switch strings.ToLower(a.Type) {
	case "emptydir":
		if len(a.DefaultMode) > 0 {
			return errors.New("--default-mode is only available for secrets and configmaps")
		}
	case "hostpath":
		if len(a.Path) == 0 {
			return errors.New("must provide --path for --type=hostPath")
		}
		if len(a.DefaultMode) > 0 {
			return errors.New("--default-mode is only available for secrets and configmaps")
		}
	case "secret":
		if len(a.SecretName) == 0 {
			return errors.New("must provide --secret-name for --type=secret")
		}
		if len(a.DefaultMode) > 0 {
			if ok, _ := regexp.MatchString(`\b0?[0-7]{3}\b`, a.DefaultMode); !ok {
				return errors.New("--default-mode must be between 0000 and 0777")
			}
		}
	case "configmap":
		if len(a.ConfigMapName) == 0 {
			return errors.New("must provide --configmap-name for --type=configmap")
		}
		if len(a.DefaultMode) > 0 {
			if ok, _ := regexp.MatchString(`\b0?[0-7]{3}\b`, a.DefaultMode); !ok {
				return errors.New("--default-mode must be between 0000 and 0777")
			}
		}
	case "persistentvolumeclaim", "pvc":
		if len(a.ClaimName) == 0 && len(a.ClaimSize) == 0 {
			return errors.New("must provide --claim-name or --claim-size (to create a new claim) for --type=pvc")
		}
		if len(a.DefaultMode) > 0 {
			return errors.New("--default-mode is only available for secrets and configmaps")
		}
	default:
		return errors.New("invalid volume type. Supported types: emptyDir, hostPath, secret, persistentVolumeClaim")
	}
	return nil
}
