# Secrets, configmaps and downwardAPI file mode bits

Author: Rodrigo Campos (@rata), Tim Hockin (@thockin)

Date: July 2016

Status: Design in progress

# Goal

Allow users to specify permission mode bits for a secret/configmap/downwardAPI
file mounted as a volume. For example, if a secret has several keys, a user
should be able to specify the permission mode bits for any file, and they may
all have different modes.

Let me say that with "permission" I only refer to the file mode here and I may
use them interchangeably. This is not about the file owners, although let me
know if you prefer to discuss that here too.


# Motivation

There is currently no way to set permissions on secret files mounted as volumes.
This can be a problem for applications that enforce files to have permissions
only for the owner (like fetchmail, ssh, pgpass file in postgres[1], etc.) and
it's just not possible to run them without changing the file mode. Also,
in-house applications may have this restriction too.

It doesn't seem totally wrong if someone wants to make a secret, that is
sensitive information, not world-readable (or group, too) as it is by default.
Although it's already in a container that is (hopefully) running only one
process and it might not be so bad. But people running more than one process in
a container asked for this too[2].

For example, my use case is that we are migrating to kubernetes, the migration
is in progress (and will take a while) and we have migrated our deployment web
interface to kubernetes. But this interface connects to the servers via ssh, so
it needs the ssh keys, and ssh will only work if the ssh key file mode is the
one it expects.

This was asked on the mailing list here[2] and here[3], too.

[1]: https://www.postgresql.org/docs/9.1/static/libpq-pgpass.html
[2]: https://groups.google.com/forum/#!topic/kubernetes-dev/eTnfMJSqmaM
[3]: https://groups.google.com/forum/#!topic/google-containers/EcaOPq4M758

# Alternatives considered

Several alternatives have been considered:

 * Add a mode to the API definition when using secrets: this is backward
   compatible as described in (docs/devel/api_changes.md) IIUC and seems like the
   way to go. Also @thockin said in the ML that he would consider such an
   approach. But it might be worth to consider if we want to do the same for
   configmaps or owners, but there is no need to do it now either.

 * Change the default file mode for secrets: I think this is unacceptable as it
   is stated in the api_changes doc. And besides it doesn't feel correct IMHO, it
   is technically one option. The argument for this might be that world and group
   readable for a secret is not a nice default, we already take care of not
   writing it to disk, etc. but the file is created world-readable anyways. Such a
   default change has been done recently: the default was 0444 in kubernetes <= 1.2
   and is now 0644 in kubernetes >= 1.3 (and the file is not a regular file,
   it's a symlink now). This change was done here to minimize differences between
   configmaps and secrets: https://github.com/kubernetes/kubernetes/pull/25285. But
   doing it again, and changing to something more restrictive (now is 0644 and it
   should be 0400 to work with ssh and most apps) seems too risky, it's even more
   restrictive than in k8s 1.2. Specially if there is no way to revert to the old
   permissions and some use case is broken by this. And if we are adding a way to
   change it, like in the option above, there is no need to rush changing the
   default. So I would discard this.

 * We don't want to people be able to change this, at least for now, and the
   ones who do, suggest that do it as a "postStart" command. This is acceptable
   if we don't want to change kubernetes core for some reason, although there
   seem to be valid use cases. But if the user want's to use the "postStart" for
   something else, then it is more disturbing to do both things (have a script
   in the docker image that deals with this, but is not probably concern of the
   project so it's not nice, or specify several commands by using "sh").

# Proposed implementation

The proposed implementation goes with the first alternative: adding a `mode`
to the API.

There will be a `defaultMode`, type `int`, in: `type SecretVolumeSource`, `type
ConfigMapVolumeSource` and `type DownwardAPIVolumeSource`. And a `mode`, type
`int` too, in `type KeyToPath` and `DownwardAPIVolumeFile`.

The mask provided in any of these fields will be ANDed with 0777 to disallow
setting sticky and setuid bits. It's not clear that use case is needed nor
really understood. And directories within the volume will be created as before
and are not affected by this setting.

In other words, the fields will look like this:

```
type SecretVolumeSource struct {
        // Name of the secret in the pod's namespace to use.
        SecretName string `json:"secretName,omitempty"`
        // If unspecified, each key-value pair in the Data field of the referenced
        // Secret will be projected into the volume as a file whose name is the
        // key and content is the value. If specified, the listed keys will be
        // projected into the specified paths, and unlisted keys will not be
        // present. If a key is specified which is not present in the Secret,
        // the volume setup will error. Paths must be relative and may not contain
        // the '..' path or start with '..'.
        Items       []KeyToPath `json:"items,omitempty"`
        // Mode bits to use on created files by default. The used mode bits will
        // be the provided AND 0777.
        // Directories within the path are not affected by this setting
        DefaultMode int32         `json:"defaultMode,omitempty"`
}

type ConfigMapVolumeSource struct {
        LocalObjectReference `json:",inline"`
        // If unspecified, each key-value pair in the Data field of the referenced
        // ConfigMap will be projected into the volume as a file whose name is the
        // key and content is the value. If specified, the listed keys will be
        // projected into the specified paths, and unlisted keys will not be
        // present. If a key is specified which is not present in the ConfigMap,
        // the volume setup will error. Paths must be relative and may not contain
        // the '..' path or start with '..'.
        Items       []KeyToPath `json:"items,omitempty"`
        // Mode bits to use on created files by default. The used mode bits will
        // be the provided AND 0777.
        // Directories within the path are not affected by this setting
        DefaultMode int32         `json:"defaultMode,omitempty"`
}

type KeyToPath struct {
        // The key to project.
        Key string `json:"key"`

        // The relative path of the file to map the key to.
        // May not be an absolute path.
        // May not contain the path element '..'.
        // May not start with the string '..'.
        Path string `json:"path"`
        // Mode bits to use on this file. The used mode bits will be the
        // provided AND 0777.
        Mode int32 `json:"mode,omitempty"`
}

type DownwardAPIVolumeSource struct {
        // Items is a list of DownwardAPIVolume file
        Items []DownwardAPIVolumeFile `json:"items,omitempty"`
        // Mode bits to use on created files by default. The used mode bits will
        // be the provided AND 0777.
        // Directories within the path are not affected by this setting
        DefaultMode int32         `json:"defaultMode,omitempty"`
}

type DownwardAPIVolumeFile struct {
        // Required: Path is  the relative path name of the file to be created. Must not be absolute or contain the '..' path. Must be utf-8 encoded. The first item of the relative path must not start with '..'
        Path string `json:"path"`
        // Required: Selects a field of the pod: only annotations, labels, name and  namespace are supported.
        FieldRef *ObjectFieldSelector `json:"fieldRef,omitempty"`
        // Selects a resource of the container: only resources limits and requests
        // (limits.cpu, limits.memory, requests.cpu and requests.memory) are currently supported.
        ResourceFieldRef *ResourceFieldSelector `json:"resourceFieldRef,omitempty"`
        // Mode bits to use on this file. The used mode bits will be the
        // provided AND 0777.
        Mode int32 `json:"mode,omitempty"`
}
```

Adding it there allows the user to change the mode bits of every file in the
object, so it achieves the goal, while having the option to have a default and
not specify all files in the object.

The are two downside:

 * The files are symlinks pointint to the real file, and the realfile
   permissions are only set. The symlink has the classic symlink permissions.
   This is something already present in 1.3, and it seems applications like ssh
   work just fine with that. Something worth mentioning, but doesn't seem to be
   an issue.
 * If the secret/configMap/downwardAPI is mounted in more than one container,
   the file permissions will be the same on all. This is already the case for
   Key mappings and doesn't seem like a big issue either.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/secret-configmap-downwarapi-file-mode.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
