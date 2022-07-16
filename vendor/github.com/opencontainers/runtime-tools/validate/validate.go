package validate

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/blang/semver"
	"github.com/hashicorp/go-multierror"
	rspec "github.com/opencontainers/runtime-spec/specs-go"
	osFilepath "github.com/opencontainers/runtime-tools/filepath"
	"github.com/sirupsen/logrus"
	"github.com/syndtr/gocapability/capability"

	"github.com/opencontainers/runtime-tools/specerror"
	"github.com/xeipuuv/gojsonschema"
)

const specConfig = "config.json"

var (
	// http://pubs.opengroup.org/onlinepubs/9699919799/functions/getrlimit.html
	posixRlimits = []string{
		"RLIMIT_AS",
		"RLIMIT_CORE",
		"RLIMIT_CPU",
		"RLIMIT_DATA",
		"RLIMIT_FSIZE",
		"RLIMIT_NOFILE",
		"RLIMIT_STACK",
	}

	// https://git.kernel.org/pub/scm/docs/man-pages/man-pages.git/tree/man2/getrlimit.2?h=man-pages-4.13
	linuxRlimits = append(posixRlimits, []string{
		"RLIMIT_MEMLOCK",
		"RLIMIT_MSGQUEUE",
		"RLIMIT_NICE",
		"RLIMIT_NPROC",
		"RLIMIT_RSS",
		"RLIMIT_RTPRIO",
		"RLIMIT_RTTIME",
		"RLIMIT_SIGPENDING",
	}...)

	configSchemaTemplate = "https://raw.githubusercontent.com/opencontainers/runtime-spec/v%s/schema/config-schema.json"
)

// Validator represents a validator for runtime bundle
type Validator struct {
	spec         *rspec.Spec
	bundlePath   string
	HostSpecific bool
	platform     string
}

// NewValidator creates a Validator
func NewValidator(spec *rspec.Spec, bundlePath string, hostSpecific bool, platform string) (Validator, error) {
	if hostSpecific && platform != runtime.GOOS {
		return Validator{}, fmt.Errorf("When hostSpecific is set, platform must be same as the host platform")
	}
	return Validator{
		spec:         spec,
		bundlePath:   bundlePath,
		HostSpecific: hostSpecific,
		platform:     platform,
	}, nil
}

// NewValidatorFromPath creates a Validator with specified bundle path
func NewValidatorFromPath(bundlePath string, hostSpecific bool, platform string) (Validator, error) {
	if bundlePath == "" {
		return Validator{}, fmt.Errorf("bundle path shouldn't be empty")
	}

	if _, err := os.Stat(bundlePath); err != nil {
		return Validator{}, err
	}

	configPath := filepath.Join(bundlePath, specConfig)
	content, err := ioutil.ReadFile(configPath)
	if err != nil {
		return Validator{}, specerror.NewError(specerror.ConfigInRootBundleDir, err, rspec.Version)
	}
	if !utf8.Valid(content) {
		return Validator{}, fmt.Errorf("%q is not encoded in UTF-8", configPath)
	}
	var spec rspec.Spec
	if err = json.Unmarshal(content, &spec); err != nil {
		return Validator{}, err
	}

	return NewValidator(&spec, bundlePath, hostSpecific, platform)
}

// CheckAll checks all parts of runtime bundle
func (v *Validator) CheckAll() error {
	var errs *multierror.Error
	errs = multierror.Append(errs, v.CheckJSONSchema())
	errs = multierror.Append(errs, v.CheckPlatform())
	errs = multierror.Append(errs, v.CheckRoot())
	errs = multierror.Append(errs, v.CheckMandatoryFields())
	errs = multierror.Append(errs, v.CheckSemVer())
	errs = multierror.Append(errs, v.CheckMounts())
	errs = multierror.Append(errs, v.CheckProcess())
	errs = multierror.Append(errs, v.CheckLinux())
	errs = multierror.Append(errs, v.CheckAnnotations())
	if v.platform == "linux" || v.platform == "solaris" {
		errs = multierror.Append(errs, v.CheckHooks())
	}

	return errs.ErrorOrNil()
}

// JSONSchemaURL returns the URL for the JSON Schema specifying the
// configuration format.  It consumes configSchemaTemplate, but we
// provide it as a function to isolate consumers from inconsistent
// naming as runtime-spec evolves.
func JSONSchemaURL(version string) (url string, err error) {
	ver, err := semver.Parse(version)
	if err != nil {
		return "", specerror.NewError(specerror.SpecVersionInSemVer, err, rspec.Version)
	}
	configRenamedToConfigSchemaVersion, err := semver.Parse("1.0.0-rc2") // config.json became config-schema.json in 1.0.0-rc2
	if ver.Compare(configRenamedToConfigSchemaVersion) == -1 {
		return "", fmt.Errorf("unsupported configuration version (older than %s)", configRenamedToConfigSchemaVersion)
	}
	return fmt.Sprintf(configSchemaTemplate, version), nil
}

// CheckJSONSchema validates the configuration against the
// runtime-spec JSON Schema, using the version of the schema that
// matches the configuration's declared version.
func (v *Validator) CheckJSONSchema() (errs error) {
	logrus.Debugf("check JSON schema")

	url, err := JSONSchemaURL(v.spec.Version)
	if err != nil {
		errs = multierror.Append(errs, err)
		return errs
	}

	schemaLoader := gojsonschema.NewReferenceLoader(url)
	documentLoader := gojsonschema.NewGoLoader(v.spec)
	result, err := gojsonschema.Validate(schemaLoader, documentLoader)
	if err != nil {
		errs = multierror.Append(errs, err)
		return errs
	}

	if !result.Valid() {
		for _, resultError := range result.Errors() {
			errs = multierror.Append(errs, errors.New(resultError.String()))
		}
	}

	return errs
}

// CheckRoot checks status of v.spec.Root
func (v *Validator) CheckRoot() (errs error) {
	logrus.Debugf("check root")

	if v.platform == "windows" && v.spec.Windows != nil {
		if v.spec.Windows.HyperV != nil {
			if v.spec.Root != nil {
				errs = multierror.Append(errs,
					specerror.NewError(specerror.RootOnHyperVNotSet, fmt.Errorf("for Hyper-V containers, Root must not be set"), rspec.Version))
			}
			return
		} else if v.spec.Root == nil {
			errs = multierror.Append(errs,
				specerror.NewError(specerror.RootOnWindowsRequired, fmt.Errorf("on Windows, for Windows Server Containers, this field is REQUIRED"), rspec.Version))
			return
		}
	} else if v.platform != "windows" && v.spec.Root == nil {
		errs = multierror.Append(errs,
			specerror.NewError(specerror.RootOnNonWindowsRequired, fmt.Errorf("on all other platforms, this field is REQUIRED"), rspec.Version))
		return
	}

	if v.platform == "windows" {
		matched, err := regexp.MatchString(`\\\\[?]\\Volume[{][a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}[}]\\`, v.spec.Root.Path)
		if err != nil {
			errs = multierror.Append(errs, err)
		} else if !matched {
			errs = multierror.Append(errs,
				specerror.NewError(specerror.RootPathOnWindowsGUID, fmt.Errorf("root.path is %q, but it MUST be a volume GUID path when target platform is windows", v.spec.Root.Path), rspec.Version))
		}

		if v.spec.Root.Readonly {
			errs = multierror.Append(errs,
				specerror.NewError(specerror.RootReadonlyOnWindowsFalse, fmt.Errorf("root.readonly field MUST be omitted or false when target platform is windows"), rspec.Version))
		}

		return
	}

	absBundlePath, err := filepath.Abs(v.bundlePath)
	if err != nil {
		errs = multierror.Append(errs, fmt.Errorf("unable to convert %q to an absolute path", v.bundlePath))
		return
	}

	if filepath.Base(v.spec.Root.Path) != "rootfs" {
		errs = multierror.Append(errs,
			specerror.NewError(specerror.RootPathOnPosixConvention, fmt.Errorf("path name should be the conventional 'rootfs'"), rspec.Version))
	}

	var rootfsPath string
	var absRootPath string
	if filepath.IsAbs(v.spec.Root.Path) {
		rootfsPath = v.spec.Root.Path
		absRootPath = filepath.Clean(rootfsPath)
	} else {
		var err error
		rootfsPath = filepath.Join(v.bundlePath, v.spec.Root.Path)
		absRootPath, err = filepath.Abs(rootfsPath)
		if err != nil {
			errs = multierror.Append(errs, fmt.Errorf("unable to convert %q to an absolute path", rootfsPath))
			return
		}
	}

	if fi, err := os.Stat(rootfsPath); err != nil {
		errs = multierror.Append(errs,
			specerror.NewError(specerror.RootPathExist, fmt.Errorf("cannot find the root path %q", rootfsPath), rspec.Version))
	} else if !fi.IsDir() {
		errs = multierror.Append(errs,
			specerror.NewError(specerror.RootPathExist, fmt.Errorf("root.path %q is not a directory", rootfsPath), rspec.Version))
	}

	rootParent := filepath.Dir(absRootPath)
	if absRootPath == string(filepath.Separator) || rootParent != absBundlePath {
		errs = multierror.Append(errs,
			specerror.NewError(specerror.ArtifactsInSingleDir, fmt.Errorf("root.path is %q, but it MUST be a child of %q", v.spec.Root.Path, absBundlePath), rspec.Version))
	}

	return
}

// CheckSemVer checks v.spec.Version
func (v *Validator) CheckSemVer() (errs error) {
	logrus.Debugf("check semver")

	version := v.spec.Version
	_, err := semver.Parse(version)
	if err != nil {
		errs = multierror.Append(errs,
			specerror.NewError(specerror.SpecVersionInSemVer, fmt.Errorf("%q is not valid SemVer: %s", version, err.Error()), rspec.Version))
	}
	if version != rspec.Version {
		errs = multierror.Append(errs, fmt.Errorf("validate currently only handles version %s, but the supplied configuration targets %s", rspec.Version, version))
	}

	return
}

// CheckHooks check v.spec.Hooks
func (v *Validator) CheckHooks() (errs error) {
	logrus.Debugf("check hooks")

	if v.platform != "linux" && v.platform != "solaris" {
		errs = multierror.Append(errs, fmt.Errorf("For %q platform, the configuration structure does not support hooks", v.platform))
		return
	}

	if v.spec.Hooks != nil {
		errs = multierror.Append(errs, v.checkEventHooks("prestart", v.spec.Hooks.Prestart, v.HostSpecific))
		errs = multierror.Append(errs, v.checkEventHooks("poststart", v.spec.Hooks.Poststart, v.HostSpecific))
		errs = multierror.Append(errs, v.checkEventHooks("poststop", v.spec.Hooks.Poststop, v.HostSpecific))
	}

	return
}

func (v *Validator) checkEventHooks(hookType string, hooks []rspec.Hook, hostSpecific bool) (errs error) {
	for i, hook := range hooks {
		if !osFilepath.IsAbs(v.platform, hook.Path) {
			errs = multierror.Append(errs,
				specerror.NewError(
					specerror.PosixHooksPathAbs,
					fmt.Errorf("hooks.%s[%d].path %v: is not absolute path",
						hookType, i, hook.Path),
					rspec.Version))
		}

		if hostSpecific {
			fi, err := os.Stat(hook.Path)
			if err != nil {
				errs = multierror.Append(errs, fmt.Errorf("cannot find %s hook: %v", hookType, hook.Path))
			}
			if fi.Mode()&0111 == 0 {
				errs = multierror.Append(errs, fmt.Errorf("the %s hook %v: is not executable", hookType, hook.Path))
			}
		}

		for _, env := range hook.Env {
			if !envValid(env) {
				errs = multierror.Append(errs, fmt.Errorf("env %q for hook %v is in the invalid form", env, hook.Path))
			}
		}
	}

	return
}

// CheckProcess checks v.spec.Process
func (v *Validator) CheckProcess() (errs error) {
	logrus.Debugf("check process")

	if v.spec.Process == nil {
		return
	}

	process := v.spec.Process
	if !osFilepath.IsAbs(v.platform, process.Cwd) {
		errs = multierror.Append(errs,
			specerror.NewError(
				specerror.ProcCwdAbs,
				fmt.Errorf("cwd %q is not an absolute path", process.Cwd),
				rspec.Version))
	}

	for _, env := range process.Env {
		if !envValid(env) {
			errs = multierror.Append(errs, fmt.Errorf("env %q should be in the form of 'key=value'. The left hand side must consist solely of letters, digits, and underscores '_'", env))
		}
	}

	if len(process.Args) == 0 {
		errs = multierror.Append(errs,
			specerror.NewError(
				specerror.ProcArgsOneEntryRequired,
				fmt.Errorf("args must not be empty"),
				rspec.Version))
	} else {
		if filepath.IsAbs(process.Args[0]) && v.spec.Root != nil {
			var rootfsPath string
			if filepath.IsAbs(v.spec.Root.Path) {
				rootfsPath = v.spec.Root.Path
			} else {
				rootfsPath = filepath.Join(v.bundlePath, v.spec.Root.Path)
			}
			absPath := filepath.Join(rootfsPath, process.Args[0])
			fileinfo, err := os.Stat(absPath)
			if os.IsNotExist(err) {
				logrus.Warnf("executable %q is not available in rootfs currently", process.Args[0])
			} else if err != nil {
				errs = multierror.Append(errs, err)
			} else {
				m := fileinfo.Mode()
				if m.IsDir() || m&0111 == 0 {
					errs = multierror.Append(errs, fmt.Errorf("arg %q is not executable", process.Args[0]))
				}
			}
		}
	}

	if v.platform == "linux" || v.platform == "solaris" {
		errs = multierror.Append(errs, v.CheckRlimits())
	}

	if v.platform == "linux" {
		if v.spec.Process.Capabilities != nil {
			errs = multierror.Append(errs, v.CheckCapabilities())
		}

		if len(process.ApparmorProfile) > 0 {
			profilePath := filepath.Join(v.bundlePath, v.spec.Root.Path, "/etc/apparmor.d", process.ApparmorProfile)
			_, err := os.Stat(profilePath)
			if err != nil {
				errs = multierror.Append(errs, err)
			}
		}
	}

	return
}

// CheckCapabilities checks v.spec.Process.Capabilities
func (v *Validator) CheckCapabilities() (errs error) {
	if v.platform != "linux" {
		errs = multierror.Append(errs, fmt.Errorf("For %q platform, the configuration structure does not support process.capabilities", v.platform))
		return
	}

	process := v.spec.Process
	var effective, permitted, inheritable, ambient bool
	caps := make(map[string][]string)

	for _, cap := range process.Capabilities.Bounding {
		caps[cap] = append(caps[cap], "bounding")
	}
	for _, cap := range process.Capabilities.Effective {
		caps[cap] = append(caps[cap], "effective")
	}
	for _, cap := range process.Capabilities.Inheritable {
		caps[cap] = append(caps[cap], "inheritable")
	}
	for _, cap := range process.Capabilities.Permitted {
		caps[cap] = append(caps[cap], "permitted")
	}
	for _, cap := range process.Capabilities.Ambient {
		caps[cap] = append(caps[cap], "ambient")
	}

	for capability, owns := range caps {
		if err := CapValid(capability, v.HostSpecific); err != nil {
			errs = multierror.Append(errs, fmt.Errorf("capability %q is not valid, man capabilities(7)", capability))
		}

		effective, permitted, ambient, inheritable = false, false, false, false
		for _, set := range owns {
			if set == "effective" {
				effective = true
				continue
			}
			if set == "inheritable" {
				inheritable = true
				continue
			}
			if set == "permitted" {
				permitted = true
				continue
			}
			if set == "ambient" {
				ambient = true
				continue
			}
		}
		if effective && !permitted {
			errs = multierror.Append(errs, fmt.Errorf("effective capability %q is not allowed, as it's not permitted", capability))
		}
		if ambient && !(permitted && inheritable) {
			errs = multierror.Append(errs, fmt.Errorf("ambient capability %q is not allowed, as it's not permitted and inheribate", capability))
		}
	}

	return
}

// CheckRlimits checks v.spec.Process.Rlimits
func (v *Validator) CheckRlimits() (errs error) {
	if v.platform != "linux" && v.platform != "solaris" {
		errs = multierror.Append(errs, fmt.Errorf("For %q platform, the configuration structure does not support process.rlimits", v.platform))
		return
	}

	process := v.spec.Process
	for index, rlimit := range process.Rlimits {
		for i := index + 1; i < len(process.Rlimits); i++ {
			if process.Rlimits[index].Type == process.Rlimits[i].Type {
				errs = multierror.Append(errs,
					specerror.NewError(
						specerror.PosixProcRlimitsErrorOnDup,
						fmt.Errorf("rlimit can not contain the same type %q",
							process.Rlimits[index].Type),
						rspec.Version))
			}
		}
		errs = multierror.Append(errs, v.rlimitValid(rlimit))
	}

	return
}

func supportedMountTypes(OS string, hostSpecific bool) (map[string]bool, error) {
	supportedTypes := make(map[string]bool)

	if OS != "linux" && OS != "windows" {
		logrus.Warnf("%v is not supported to check mount type", OS)
		return nil, nil
	} else if OS == "windows" {
		supportedTypes["ntfs"] = true
		return supportedTypes, nil
	}

	if hostSpecific {
		f, err := os.Open("/proc/filesystems")
		if err != nil {
			return nil, err
		}
		defer f.Close()

		s := bufio.NewScanner(f)
		for s.Scan() {
			if err := s.Err(); err != nil {
				return supportedTypes, err
			}

			text := s.Text()
			parts := strings.Split(text, "\t")
			if len(parts) > 1 {
				supportedTypes[parts[1]] = true
			} else {
				supportedTypes[parts[0]] = true
			}
		}

		supportedTypes["bind"] = true

		return supportedTypes, nil
	}
	logrus.Warn("Checking linux mount types without --host-specific is not supported yet")
	return nil, nil
}

// CheckMounts checks v.spec.Mounts
func (v *Validator) CheckMounts() (errs error) {
	logrus.Debugf("check mounts")

	supportedTypes, err := supportedMountTypes(v.platform, v.HostSpecific)
	if err != nil {
		errs = multierror.Append(errs, err)
		return
	}

	for i, mountA := range v.spec.Mounts {
		if supportedTypes != nil && !supportedTypes[mountA.Type] {
			errs = multierror.Append(errs, fmt.Errorf("unsupported mount type %q", mountA.Type))
		}
		if !osFilepath.IsAbs(v.platform, mountA.Destination) {
			errs = multierror.Append(errs,
				specerror.NewError(
					specerror.MountsDestAbs,
					fmt.Errorf("mounts[%d].destination %q is not absolute",
						i,
						mountA.Destination),
					rspec.Version))
		}
		for j, mountB := range v.spec.Mounts {
			if i == j {
				continue
			}
			// whether B.Desination is nested within A.Destination
			nested, err := osFilepath.IsAncestor(v.platform, mountA.Destination, mountB.Destination, ".")
			if err != nil {
				errs = multierror.Append(errs, err)
				continue
			}
			if nested {
				if v.platform == "windows" && i < j {
					errs = multierror.Append(errs,
						specerror.NewError(
							specerror.MountsDestOnWindowsNotNested,
							fmt.Errorf("on Windows, %v nested within %v is forbidden",
								mountB.Destination, mountA.Destination),
							rspec.Version))
				}
				if i > j {
					logrus.Warnf("%v will be covered by %v", mountB.Destination, mountA.Destination)
				}
			}
		}
	}

	return
}

// CheckPlatform checks v.platform
func (v *Validator) CheckPlatform() (errs error) {
	logrus.Debugf("check platform")

	if v.platform != "linux" && v.platform != "solaris" && v.platform != "windows" {
		errs = multierror.Append(errs, fmt.Errorf("platform %q is not supported", v.platform))
		return
	}

	if v.HostSpecific && v.platform != runtime.GOOS {
		errs = multierror.Append(errs, fmt.Errorf("platform %q differs from the host %q, skipping host-specific checks", v.platform, runtime.GOOS))
		v.HostSpecific = false
	}

	if v.platform == "windows" {
		if v.spec.Windows == nil {
			errs = multierror.Append(errs,
				specerror.NewError(
					specerror.PlatformSpecConfOnWindowsSet,
					fmt.Errorf("'windows' MUST be set when platform is `windows`"),
					rspec.Version))
		}
	}

	return
}

// CheckLinuxResources checks v.spec.Linux.Resources
func (v *Validator) CheckLinuxResources() (errs error) {
	logrus.Debugf("check linux resources")

	r := v.spec.Linux.Resources
	if r.Memory != nil {
		if r.Memory.Limit != nil && r.Memory.Swap != nil && uint64(*r.Memory.Limit) > uint64(*r.Memory.Swap) {
			errs = multierror.Append(errs, fmt.Errorf("minimum memoryswap should be larger than memory limit"))
		}
		if r.Memory.Limit != nil && r.Memory.Reservation != nil && uint64(*r.Memory.Reservation) > uint64(*r.Memory.Limit) {
			errs = multierror.Append(errs, fmt.Errorf("minimum memory limit should be larger than memory reservation"))
		}
	}
	if r.Network != nil && v.HostSpecific {
		var exist bool
		interfaces, err := net.Interfaces()
		if err != nil {
			errs = multierror.Append(errs, err)
			return
		}
		for _, prio := range r.Network.Priorities {
			exist = false
			for _, ni := range interfaces {
				if prio.Name == ni.Name {
					exist = true
					break
				}
			}
			if !exist {
				errs = multierror.Append(errs, fmt.Errorf("interface %s does not exist currently", prio.Name))
			}
		}
	}
	for index := 0; index < len(r.Devices); index++ {
		switch r.Devices[index].Type {
		case "a", "b", "c", "":
		default:
			errs = multierror.Append(errs, fmt.Errorf("type of devices %s is invalid", r.Devices[index].Type))
		}

		access := []byte(r.Devices[index].Access)
		for i := 0; i < len(access); i++ {
			switch access[i] {
			case 'r', 'w', 'm':
			default:
				errs = multierror.Append(errs, fmt.Errorf("access %s is invalid", r.Devices[index].Access))
				return
			}
		}
	}

	if r.BlockIO != nil && r.BlockIO.WeightDevice != nil {
		for i, weightDevice := range r.BlockIO.WeightDevice {
			if weightDevice.Weight == nil && weightDevice.LeafWeight == nil {
				errs = multierror.Append(errs,
					specerror.NewError(
						specerror.BlkIOWeightOrLeafWeightExist,
						fmt.Errorf("linux.resources.blockIO.weightDevice[%d] specifies neither weight nor leafWeight", i),
						rspec.Version))
			}
		}
	}

	return
}

// CheckAnnotations checks v.spec.Annotations
func (v *Validator) CheckAnnotations() (errs error) {
	logrus.Debugf("check annotations")

	reversedDomain := regexp.MustCompile(`^[A-Za-z]{2,6}(\.[A-Za-z0-9-]{1,63})+$`)
	for key := range v.spec.Annotations {
		if strings.HasPrefix(key, "org.opencontainers") {
			errs = multierror.Append(errs,
				specerror.NewError(
					specerror.AnnotationsKeyReservedNS,
					fmt.Errorf("key %q is reserved", key),
					rspec.Version))
		}

		if !reversedDomain.MatchString(key) {
			errs = multierror.Append(errs,
				specerror.NewError(
					specerror.AnnotationsKeyReversedDomain,
					fmt.Errorf("key %q SHOULD be named using a reverse domain notation", key),
					rspec.Version))
		}
	}

	return
}

// CapValid checks whether a capability is valid
func CapValid(c string, hostSpecific bool) error {
	isValid := false

	if !strings.HasPrefix(c, "CAP_") {
		return fmt.Errorf("capability %s must start with CAP_", c)
	}
	for _, cap := range capability.List() {
		if c == fmt.Sprintf("CAP_%s", strings.ToUpper(cap.String())) {
			if hostSpecific && cap > LastCap() {
				return fmt.Errorf("%s is not supported on the current host", c)
			}
			isValid = true
			break
		}
	}

	if !isValid {
		return fmt.Errorf("invalid capability: %s", c)
	}
	return nil
}

func envValid(env string) bool {
	items := strings.Split(env, "=")
	if len(items) < 2 {
		return false
	}
	for i, ch := range strings.TrimSpace(items[0]) {
		if !unicode.IsDigit(ch) && !unicode.IsLetter(ch) && ch != '_' {
			return false
		}
		if i == 0 && unicode.IsDigit(ch) {
			logrus.Warnf("Env %v: variable name beginning with digit is not recommended.", env)
		}
	}
	return true
}

func (v *Validator) rlimitValid(rlimit rspec.POSIXRlimit) (errs error) {
	if rlimit.Hard < rlimit.Soft {
		errs = multierror.Append(errs, fmt.Errorf("hard limit of rlimit %s should not be less than soft limit", rlimit.Type))
	}

	if v.platform == "linux" {
		for _, val := range linuxRlimits {
			if val == rlimit.Type {
				return
			}
		}
		errs = multierror.Append(errs, specerror.NewError(specerror.PosixProcRlimitsTypeValueError, fmt.Errorf("rlimit type %q may not be valid", rlimit.Type), v.spec.Version))
	} else if v.platform == "solaris" {
		for _, val := range posixRlimits {
			if val == rlimit.Type {
				return
			}
		}
		errs = multierror.Append(errs, specerror.NewError(specerror.PosixProcRlimitsTypeValueError, fmt.Errorf("rlimit type %q may not be valid", rlimit.Type), v.spec.Version))
	} else {
		logrus.Warnf("process.rlimits validation not yet implemented for platform %q", v.platform)
	}

	return
}

func isStruct(t reflect.Type) bool {
	return t.Kind() == reflect.Struct
}

func isStructPtr(t reflect.Type) bool {
	return t.Kind() == reflect.Ptr && t.Elem().Kind() == reflect.Struct
}

func checkMandatoryUnit(field reflect.Value, tagField reflect.StructField, parent string) (errs error) {
	mandatory := !strings.Contains(tagField.Tag.Get("json"), "omitempty")
	switch field.Kind() {
	case reflect.Ptr:
		if mandatory && field.IsNil() {
			errs = multierror.Append(errs, fmt.Errorf("'%s.%s' should not be empty", parent, tagField.Name))
		}
	case reflect.String:
		if mandatory && (field.Len() == 0) {
			errs = multierror.Append(errs, fmt.Errorf("'%s.%s' should not be empty", parent, tagField.Name))
		}
	case reflect.Slice:
		if mandatory && (field.IsNil() || field.Len() == 0) {
			errs = multierror.Append(errs, fmt.Errorf("'%s.%s' should not be empty", parent, tagField.Name))
			return
		}
		for index := 0; index < field.Len(); index++ {
			mValue := field.Index(index)
			if mValue.CanInterface() {
				errs = multierror.Append(errs, checkMandatory(mValue.Interface()))
			}
		}
	case reflect.Map:
		if mandatory && (field.IsNil() || field.Len() == 0) {
			errs = multierror.Append(errs, fmt.Errorf("'%s.%s' should not be empty", parent, tagField.Name))
			return
		}
		keys := field.MapKeys()
		for index := 0; index < len(keys); index++ {
			mValue := field.MapIndex(keys[index])
			if mValue.CanInterface() {
				errs = multierror.Append(errs, checkMandatory(mValue.Interface()))
			}
		}
	default:
	}

	return
}

func checkMandatory(obj interface{}) (errs error) {
	objT := reflect.TypeOf(obj)
	objV := reflect.ValueOf(obj)
	if isStructPtr(objT) {
		objT = objT.Elem()
		objV = objV.Elem()
	} else if !isStruct(objT) {
		return
	}

	for i := 0; i < objT.NumField(); i++ {
		t := objT.Field(i).Type
		if isStructPtr(t) && objV.Field(i).IsNil() {
			if !strings.Contains(objT.Field(i).Tag.Get("json"), "omitempty") {
				errs = multierror.Append(errs, fmt.Errorf("'%s.%s' should not be empty", objT.Name(), objT.Field(i).Name))
			}
		} else if (isStruct(t) || isStructPtr(t)) && objV.Field(i).CanInterface() {
			errs = multierror.Append(errs, checkMandatory(objV.Field(i).Interface()))
		} else {
			errs = multierror.Append(errs, checkMandatoryUnit(objV.Field(i), objT.Field(i), objT.Name()))
		}

	}
	return
}

// CheckMandatoryFields checks mandatory field of container's config file
func (v *Validator) CheckMandatoryFields() error {
	logrus.Debugf("check mandatory fields")

	if v.spec == nil {
		return fmt.Errorf("Spec can't be nil")
	}

	return checkMandatory(v.spec)
}
