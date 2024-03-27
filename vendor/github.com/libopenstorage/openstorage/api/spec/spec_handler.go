package spec

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/pkg/parser"
	"github.com/libopenstorage/openstorage/pkg/units"
)

// SpecHandler provides conversion function from what gets passed in over the
// plugin API to an api.VolumeSpec object.
type SpecHandler interface {
	// SpecOptsFromString parses options from the name and returns in a map.
	// The input string should have known keys in the following format:
	// "scale=value;size=value;name=volname"
	// If the spec was parsed, it returns:
	//   (true, options_map, parsed_name)
	// If the input string didn't contain the name, it returns:
	//   (false, nil, inputString)
	SpecOptsFromString(inputString string) (
		bool,
		map[string]string,
		string,
	)

	// SpecFromString parses options from the name.
	// If the scheduler was unable to pass in the volume spec via the API,
	// the spec can be passed in via the name in the format:
	// "key=value;key=value;name=volname"
	// source is populated if key parent=<volume_id> is specified.
	// If the spec was parsed, it returns:
	//  	(true, parsed_spec, locator, source, parsed_name)
	// If the input string didn't contain the string, it returns:
	// 	(false, DefaultSpec(), nil, nil, inputString)
	SpecFromString(inputString string) (
		bool,
		*api.VolumeSpec,
		*api.VolumeLocator,
		*api.Source,
		string,
	)

	// GetTokenFromString parses the token from the name.
	// If the token is not present in the name, it will
	// check inside of the docker options passed in.
	// If the token was parsed, it returns:
	// 	(token, true)
	// If the token wasn't parsed, it returns:
	// 	("", false)
	GetTokenFromString(str string) (string, bool)

	// GetTokenSecretContextFromString parses the full token secret request from the name.
	// If the token secret was parsed, it returns:
	// 	(tokenSecretContext, true)
	// If the token secret wasn't parsed, it returns:
	// 	(nil, false)
	GetTokenSecretContextFromString(str string) (*api.TokenSecretContext, bool)

	// SpecFromOpts parses in docker options passed in the the docker run
	// command of the form --opt name=value
	// source is populated if --opt parent=<volume_id> is specified.
	// If the options are validated then it returns:
	// 	(resultant_VolumeSpec, source, locator, nil)
	// If the options have invalid values then it returns:
	//	(nil, nil, nil, error)
	SpecFromOpts(opts map[string]string) (
		*api.VolumeSpec,
		*api.VolumeLocator,
		*api.Source,
		error,
	)

	// UpdateSpecFromOpts parses in volume options passed through the opts map and updates given spec, locator & source
	// If the options are validated then it returns:
	// 	(resultant_VolumeSpec, source, locator, nil)
	// If the options have invalid values then it returns:
	//	(nil, nil, nil, error)
	UpdateSpecFromOpts(opts map[string]string, spec *api.VolumeSpec, locator *api.VolumeLocator, source *api.Source) (
		*api.VolumeSpec,
		*api.VolumeLocator,
		*api.Source,
		error,
	)

	// Returns a default VolumeSpec if no docker options or string encoding
	// was provided.
	DefaultSpec() *api.VolumeSpec
}

var (
	nameRegex                     = regexp.MustCompile(api.Name + "=([0-9A-Za-z_-]+),?")
	tokenRegex                    = regexp.MustCompile(api.Token + "=([A-Za-z0-9-_=]+\\.[A-Za-z0-9-_=]+\\.?[A-Za-z0-9-_.+/=]+),?")
	tokenSecretRegex              = regexp.MustCompile(api.TokenSecret + `=/*([0-9A-Za-z_/]+),?`)
	tokenSecretNamespaceRegex     = regexp.MustCompile(api.TokenSecretNamespace + `=/*([0-9A-Za-z_/]+),?`)
	nodesRegex                    = regexp.MustCompile(api.SpecNodes + "=([A-Za-z0-9-_;]+),?")
	parentRegex                   = regexp.MustCompile(api.SpecParent + "=([A-Za-z]+),?")
	sizeRegex                     = regexp.MustCompile(api.SpecSize + "=([0-9A-Za-z]+),?")
	scaleRegex                    = regexp.MustCompile(api.SpecScale + "=([0-9]+),?")
	fsRegex                       = regexp.MustCompile(api.SpecFilesystem + "=([0-9A-Za-z]+),?")
	bsRegex                       = regexp.MustCompile(api.SpecBlockSize + "=([0-9]+),?")
	queueDepthRegex               = regexp.MustCompile(api.SpecQueueDepth + "=([0-9]+),?")
	haRegex                       = regexp.MustCompile(api.SpecHaLevel + "=([0-9]+),?")
	cosRegex                      = regexp.MustCompile(api.SpecPriority + "=([A-Za-z]+),?")
	sharedRegex                   = regexp.MustCompile(api.SpecShared + "=([A-Za-z]+),?")
	journalRegex                  = regexp.MustCompile(api.SpecJournal + "=([A-Za-z]+),?")
	sharedv4Regex                 = regexp.MustCompile(api.SpecSharedv4 + "=([A-Za-z]+),?")
	cascadedRegex                 = regexp.MustCompile(api.SpecCascaded + "=([A-Za-z]+),?")
	passphraseRegex               = regexp.MustCompile(api.SpecPassphrase + "=([0-9A-Za-z_@./#&+-]+),?")
	stickyRegex                   = regexp.MustCompile(api.SpecSticky + "=([A-Za-z]+),?")
	secureRegex                   = regexp.MustCompile(api.SpecSecure + "=([A-Za-z]+),?")
	zonesRegex                    = regexp.MustCompile(api.SpecZones + "=([A-Za-z]+),?")
	racksRegex                    = regexp.MustCompile(api.SpecRacks + "=([A-Za-z]+),?")
	rackRegex                     = regexp.MustCompile(api.SpecRack + "=([A-Za-z]+),?")
	aggrRegex                     = regexp.MustCompile(api.SpecAggregationLevel + "=([0-9]+|" + api.SpecAutoAggregationValue + "),?")
	compressedRegex               = regexp.MustCompile(api.SpecCompressed + "=([A-Za-z]+),?")
	snapScheduleRegex             = regexp.MustCompile(api.SpecSnapshotSchedule + `=([A-Za-z0-9:;@=#]+),?`)
	ioProfileRegex                = regexp.MustCompile(api.SpecIoProfile + "=([0-9A-Za-z_-]+),?")
	asyncIoRegex                  = regexp.MustCompile(api.SpecAsyncIo + "=([A-Za-z]+),?")
	earlyAckRegex                 = regexp.MustCompile(api.SpecEarlyAck + "=([A-Za-z]+),?")
	forceUnsupportedFsTypeRegex   = regexp.MustCompile(api.SpecForceUnsupportedFsType + "=([A-Za-z]+),?")
	nodiscardRegex                = regexp.MustCompile(api.SpecNodiscard + "=([A-Za-z]+),?")
	storagePolicyRegex            = regexp.MustCompile(api.StoragePolicy + "=([0-9A-Za-z_-]+),?")
	exportProtocolRegex           = regexp.MustCompile(api.SpecExportProtocol + "=([A-Za-z]+),?")
	exportOptionsRegex            = regexp.MustCompile(api.SpecExportOptions + "=([A-Za-z]+),?")
	proxyEndpointRegex            = regexp.MustCompile(api.SpecProxyEndpoint + "=([0-9A-Za-z_@:./#&+-]+),?")
	proxyNFSSubPathRegex          = regexp.MustCompile(api.SpecProxyNFSSubPath + "=([A-Za-z]+),?")
	proxyNFSExportPathRegex       = regexp.MustCompile(api.SpecProxyNFSExportPath + "=([A-Za-z]+),?")
	proxyS3BucketRegex            = regexp.MustCompile(api.SpecProxyS3Bucket + "=([A-Za-z]+),?")
	mountOptionsRegex             = regexp.MustCompile(api.SpecMountOptions + `=([A-Za-z0-9:;@=#]+),?`)
	sharedv4MountOptionsRegex     = regexp.MustCompile(api.SpecSharedv4MountOptions + `=([A-Za-z0-9:;@=#]+),?`)
	cowOnDemandRegex              = regexp.MustCompile(api.SpecCowOnDemand + "=([A-Za-z]+),?")
	directIoRegex                 = regexp.MustCompile(api.SpecDirectIo + "=([A-Za-z]+),?")
	ProxyWriteRegex               = regexp.MustCompile(api.SpecProxyWrite + "=([A-Za-z]+),?")
	fastpathRegex                 = regexp.MustCompile(api.SpecFastpath + "=([A-Za-z]+),?")
	sharedv4ServiceTypeRegex      = regexp.MustCompile(api.SpecSharedv4ServiceType + "=([A-Za-z]+),?")
	sharedv4ServiceNameRegex      = regexp.MustCompile(api.SpecSharedv4ServiceName + "=([A-Za-z]+),?")
	Sharedv4FailoverStrategyRegex = regexp.MustCompile(api.SpecSharedv4FailoverStrategy + "=([A-Za-z]+),?")
	AutoFstrimRegex               = regexp.MustCompile(api.SpecAutoFstrim + "=([A-Za-z]+),?")
	SpecIoThrottleRdIOPSRegex     = regexp.MustCompile(api.SpecIoThrottleRdIOPS + "=([0-9]+),?")
	SpecIoThrottleWrIOPSRegex     = regexp.MustCompile(api.SpecIoThrottleWrIOPS + "=([0-9]+),?")
	SpecIoThrottleRdBWRegex       = regexp.MustCompile(api.SpecIoThrottleRdBW + "=([0-9]+),?")
	SpecIoThrottleWrBWRegex       = regexp.MustCompile(api.SpecIoThrottleWrBW + "=([0-9]+),?")
)

type specHandler struct {
}

// NewSpecHandler returns a new SpecHandler interface
func NewSpecHandler() SpecHandler {
	return &specHandler{}
}

func (d *specHandler) cosLevel(cos string) (api.CosType, error) {
	cos = strings.ToLower(cos)
	switch cos {
	case "high", "3":
		return api.CosType_HIGH, nil
	case "medium", "2":
		return api.CosType_MEDIUM, nil
	case "low", "1", "":
		return api.CosType_LOW, nil
	}
	return api.CosType_NONE,
		fmt.Errorf("Cos must be one of %q | %q | %q", "high", "medium", "low")
}

func (d *specHandler) getVal(r *regexp.Regexp, str string) (bool, string) {
	found := r.FindString(str)
	if found == "" {
		return false, ""
	}

	submatches := r.FindStringSubmatch(str)
	if len(submatches) < 2 {
		return false, ""
	}

	return true, submatches[1]
}

func (d *specHandler) DefaultSpec() *api.VolumeSpec {
	return &api.VolumeSpec{
		Format:    api.FSType_FS_TYPE_EXT4,
		HaLevel:   1,
		IoProfile: api.IoProfile_IO_PROFILE_AUTO,
		Xattr:     api.Xattr_COW_ON_DEMAND,
	}
}

func (d *specHandler) UpdateSpecFromOpts(opts map[string]string, spec *api.VolumeSpec, locator *api.VolumeLocator,
	source *api.Source) (*api.VolumeSpec, *api.VolumeLocator, *api.Source, error) {
	nodeList := make([]string, 0)

	if spec == nil {
		spec = d.DefaultSpec()
	}

	if source == nil {
		source = &api.Source{}
	}

	if locator == nil {
		locator = &api.VolumeLocator{
			VolumeLabels: make(map[string]string),
		}
	}

	// Since we don't know which kind of Pure proxy spec this will be, we set this in
	// a variable outside the loop and then set it in the proper field at the end
	var pureBackendVolName *string

	for k, v := range opts {
		switch k {
		case api.SpecNodes:
			inputNodes := strings.Split(strings.Replace(v, ";", ",", -1), ",")
			for _, node := range inputNodes {
				if len(node) != 0 {
					nodeList = append(nodeList, node)
				}
			}
			spec.ReplicaSet = &api.ReplicaSet{Nodes: nodeList}
		case api.SpecParent:
			source.Parent = v
		case api.SpecEphemeral:
			spec.Ephemeral, _ = strconv.ParseBool(v)
		case api.SpecSize:
			if size, err := units.Parse(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Size = uint64(size)
			}
		case api.SpecScale:
			if scale, err := strconv.ParseUint(v, 10, 64); err == nil {
				spec.Scale = uint32(scale)
			}

		case api.SpecFilesystem:
			if value, err := api.FSTypeSimpleValueOf(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Format = value
			}
		case api.SpecBlockSize:
			if blockSize, err := units.Parse(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.BlockSize = blockSize
			}
		case api.SpecQueueDepth:
			if queueDepth, err := strconv.ParseInt(v, 10, 64); err != nil {
				return nil, nil, nil, err
			} else {
				spec.QueueDepth = uint32(queueDepth)
			}
		case api.SpecHaLevel:
			haLevel, _ := strconv.ParseInt(v, 10, 64)
			spec.HaLevel = haLevel
		case api.SpecPriority:
			cos, err := d.cosLevel(v)
			if err != nil {
				return nil, nil, nil, err
			}
			spec.Cos = cos
		case api.SpecPriorityAlias:
			cos, err := d.cosLevel(v)
			if err != nil {
				return nil, nil, nil, err
			}
			spec.Cos = cos
		case api.SpecDedupe:
			spec.Dedupe, _ = strconv.ParseBool(v)
		case api.SpecSnapshotInterval:
			snapshotInterval, _ := strconv.ParseUint(v, 10, 32)
			spec.SnapshotInterval = uint32(snapshotInterval)
		case api.SpecSnapshotSchedule:
			spec.SnapshotSchedule = v
		case api.SpecAggregationLevel:
			if v == api.SpecAutoAggregationValue {
				spec.AggregationLevel = api.AutoAggregation
			} else {
				aggregationLevel, _ := strconv.ParseUint(v, 10, 32)
				spec.AggregationLevel = uint32(aggregationLevel)
			}
		case api.SpecShared:
			if shared, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Shared = shared
			}
		case api.SpecJournal:
			if journal, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Journal = journal
			}
		case api.SpecSharedv4:
			if sharedv4, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Sharedv4 = sharedv4
			}
		case api.SpecCascaded:
			if cascaded, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Cascaded = cascaded
			}
		case api.SpecSticky:
			if sticky, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Sticky = sticky
			}
		case api.SpecSecure:
			if secure, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Encrypted = secure
			}
		case api.SpecPassphrase:
			spec.Encrypted = true
			spec.Passphrase = v
		case api.SpecGroup:
			spec.Group = &api.Group{Id: v}
		case api.SpecGroupEnforce:
			if groupEnforced, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.GroupEnforced = groupEnforced
			}
		case api.SpecZones, api.SpecRacks:
			locator.VolumeLabels[k] = v
		case api.SpecRack:
			locator.VolumeLabels[api.SpecRacks] = v
		case api.SpecBestEffortLocationProvisioning:
			locator.VolumeLabels[k] = "true"
		case api.SpecCompressed:
			if compressed, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Compressed = compressed
			}
		case api.SpecLabels:
			if labels, err := parser.LabelsFromString(v); err != nil {
				return nil, nil, nil, err
			} else {
				for k, v := range labels {
					locator.VolumeLabels[k] = v
				}
			}
		case api.SpecIoProfile:
			if ioProfile, err := api.IoProfileSimpleValueOf(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.IoProfile = ioProfile
			}
		case api.SpecEarlyAck:
			if earlyAck, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				if spec.IoStrategy == nil {
					spec.IoStrategy = &api.IoStrategy{}
				}
				spec.IoStrategy.EarlyAck = earlyAck
			}
		case api.SpecAsyncIo:
			if asyncIo, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				if spec.IoStrategy == nil {
					spec.IoStrategy = &api.IoStrategy{}
				}
				spec.IoStrategy.AsyncIo = asyncIo
			}
		case api.SpecForceUnsupportedFsType:
			if forceFs, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.ForceUnsupportedFsType = forceFs
			}
		case api.SpecNodiscard:
			if nodiscard, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Nodiscard = nodiscard
				// autofstrim to follow nodiscard when not explicitly specified
				if _, exists := opts[api.SpecAutoFstrim]; !exists {
					spec.AutoFstrim = nodiscard
				}
			}
		case api.Token:
			// skip, if not it would be added to the labels
		case api.StoragePolicy:
			spec.StoragePolicy = v
		case api.SpecExportProtocol:
			if spec.ExportSpec == nil {
				spec.ExportSpec = &api.ExportSpec{}
			}
			if v == api.SpecExportProtocolPXD {
				spec.ExportSpec.ExportProtocol = api.ExportProtocol_PXD
			} else if v == api.SpecExportProtocolNFS {
				spec.ExportSpec.ExportProtocol = api.ExportProtocol_NFS
			} else if v == api.SpecExportProtocolISCSI {
				spec.ExportSpec.ExportProtocol = api.ExportProtocol_ISCSI
			} else if v == api.SpecExportProtocolCustom {
				spec.ExportSpec.ExportProtocol = api.ExportProtocol_CUSTOM
			} else {
				return nil, nil, nil, fmt.Errorf("unsupported Export Protocol %v", v)
			}
		case api.SpecExportOptions:
			if spec.ExportSpec == nil {
				spec.ExportSpec = &api.ExportSpec{}
			}
			spec.ExportSpec.ExportOptions = v
		case api.SpecProxyEndpoint:
			if spec.ProxySpec == nil {
				spec.ProxySpec = &api.ProxySpec{}
			}
			proxyProtocol, endpoint := api.ParseProxyEndpoint(v)
			if proxyProtocol == api.ProxyProtocol_PROXY_PROTOCOL_INVALID {
				return nil, nil, nil, fmt.Errorf("invalid proxy endpoint: %v", v)
			}
			spec.ProxySpec.Endpoint = endpoint
			spec.ProxySpec.ProxyProtocol = proxyProtocol

		case api.SpecProxyNFSSubPath:
			if spec.ProxySpec == nil {
				spec.ProxySpec = &api.ProxySpec{}
			}
			if spec.ProxySpec.NfsSpec == nil {
				spec.ProxySpec.NfsSpec = &api.NFSProxySpec{
					SubPath: v,
				}
			} else {
				spec.ProxySpec.NfsSpec.SubPath = v
			}

		case api.SpecProxyNFSExportPath:
			if spec.ProxySpec == nil {
				spec.ProxySpec = &api.ProxySpec{}
			}
			if spec.ProxySpec.NfsSpec == nil {
				spec.ProxySpec.NfsSpec = &api.NFSProxySpec{
					ExportPath: v,
				}
			} else {
				spec.ProxySpec.NfsSpec.ExportPath = v
			}
		case api.SpecSharedv4ServiceType:
			if spec.Sharedv4ServiceSpec == nil {
				spec.Sharedv4ServiceSpec = &api.Sharedv4ServiceSpec{}
			}
			if v == "" {
				// force sharedv4 non-service
				spec.Sharedv4ServiceSpec.Type = api.Sharedv4ServiceSpec_NONE
			} else if v == "NodePort" || v == "nodePort" {
				spec.Sharedv4ServiceSpec.Type = api.Sharedv4ServiceSpec_NODEPORT
			} else if v == "ClusterIP" || v == "clusterIP" {
				spec.Sharedv4ServiceSpec.Type = api.Sharedv4ServiceSpec_CLUSTERIP
			} else if v == "LoadBalancer" || v == "loadBalancer" {
				spec.Sharedv4ServiceSpec.Type = api.Sharedv4ServiceSpec_LOADBALANCER
			} else {
				return nil, nil, nil, fmt.Errorf("invalid sharedv4 service type: %v", v)
			}
		case api.SpecSharedv4ServiceName:
			if spec.Sharedv4ServiceSpec == nil {
				spec.Sharedv4ServiceSpec = &api.Sharedv4ServiceSpec{}
			}
			spec.Sharedv4ServiceSpec.Name = v
		case api.SpecSharedv4FailoverStrategy:
			if spec.Sharedv4Spec == nil {
				spec.Sharedv4Spec = &api.Sharedv4Spec{}
			}
			if v == api.SpecSharedv4FailoverStrategyAggressive {
				spec.Sharedv4Spec.FailoverStrategy = api.Sharedv4FailoverStrategy_AGGRESSIVE
			} else if v == api.SpecSharedv4FailoverStrategyNormal {
				spec.Sharedv4Spec.FailoverStrategy = api.Sharedv4FailoverStrategy_NORMAL
			} else if v == api.SpecSharedv4FailoverStrategyUnspecified {
				spec.Sharedv4Spec.FailoverStrategy = api.Sharedv4FailoverStrategy_UNSPECIFIED
			} else {
				return nil, nil, nil, fmt.Errorf("invalid sharedv4 failover strategy: %v", v)
			}
		case api.SpecMountOptions:
			if spec.MountOptions == nil {
				spec.MountOptions = &api.MountOptions{
					Options: make(map[string]string),
				}
			}
			options, err := parser.LabelsFromString(v)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("invalid mount options format %v", v)
			}
			spec.MountOptions.Options = options
		case api.SpecFaCreateOptions:
			spec.FaCreateOptions = v
		case api.SpecSharedv4MountOptions:
			if spec.Sharedv4MountOptions == nil {
				spec.Sharedv4MountOptions = &api.MountOptions{
					Options: make(map[string]string),
				}
			}
			options, err := parser.LabelsFromString(v)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("invalid sharedv4 client mount options format %v", v)
			}
			spec.Sharedv4MountOptions.Options = options
		case api.SpecCowOnDemand:
			if cowOnDemand, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				if cowOnDemand {
					spec.Xattr = api.Xattr_COW_ON_DEMAND
				} else if spec.Xattr == api.Xattr_COW_ON_DEMAND {
					spec.Xattr = api.Xattr_UNSPECIFIED
				}
			}
		case api.SpecDirectIo:
			if directIo, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				if spec.IoStrategy == nil {
					spec.IoStrategy = &api.IoStrategy{}
				}
				spec.IoStrategy.DirectIo = directIo
			}
		case api.SpecScanPolicyTrigger:
			if spec.ScanPolicy == nil {
				spec.ScanPolicy = &api.ScanPolicy{}
			}
			if scanTrigger, err := api.ScanPolicy_ScanTriggerSimpleValueOf(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.ScanPolicy.Trigger = scanTrigger
			}
		case api.SpecScanPolicyAction:
			if spec.ScanPolicy == nil {
				spec.ScanPolicy = &api.ScanPolicy{}
			}
			if scanAction, err := api.ScanPolicy_ScanActionSimpleValueOf(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.ScanPolicy.Action = scanAction
			}
		case api.SpecProxyWrite:
			if proxyWrite, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.ProxyWrite = proxyWrite
			}
		case api.SpecFastpath:
			if fastpath, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.FpPreference = fastpath
			}
		case api.SpecBackendType:
			// Treat Pure FlashArray and FlashBlade volumes as proxy volumes and store the backend type as ProxyProtocol
			// in the spec, but the key to pass this value in is specified as 'backend' to reduce confusions
			if backendType, err := api.ProxyProtocolSimpleValueOf(v); err != nil {
				return nil, nil, nil, err
			} else {
				if spec.ProxySpec == nil {
					spec.ProxySpec = &api.ProxySpec{}
				}
				spec.ProxySpec.ProxyProtocol = backendType
			}
		case api.SpecBackendVolName:
			volName := v
			pureBackendVolName = &volName
		case api.SpecPureFileExportRules:
			if spec.ProxySpec == nil {
				spec.ProxySpec = &api.ProxySpec{}
			}
			if spec.ProxySpec.PureFileSpec == nil {
				spec.ProxySpec.PureFileSpec = &api.PureFileSpec{}
			}
			spec.ProxySpec.PureFileSpec.ExportRules = v
		case api.SpecAutoFstrim:
			if autoFstrim, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.AutoFstrim = autoFstrim
			}
		case api.SpecIoThrottleRdIOPS:
			if spec.IoThrottle == nil {
				spec.IoThrottle = &api.IoThrottle{}
			}
			if throttleIOPS, err := strconv.ParseUint(v, 10, 64); err != nil {
				return nil, nil, nil, err
			} else {
				spec.IoThrottle.ReadIops = uint32(throttleIOPS)
			}
		case api.SpecIoThrottleWrIOPS:
			if spec.IoThrottle == nil {
				spec.IoThrottle = &api.IoThrottle{}
			}
			if throttleIOPS, err := strconv.ParseUint(v, 10, 64); err != nil {
				return nil, nil, nil, err
			} else {
				spec.IoThrottle.WriteIops = uint32(throttleIOPS)
			}
		case api.SpecIoThrottleRdBW:
			if spec.IoThrottle == nil {
				spec.IoThrottle = &api.IoThrottle{}
			}
			if throttleBW, err := strconv.ParseUint(v, 10, 64); err != nil {
				return nil, nil, nil, err
			} else {
				spec.IoThrottle.ReadBwMbytes = uint32(throttleBW)
			}
		case api.SpecIoThrottleWrBW:
			if spec.IoThrottle == nil {
				spec.IoThrottle = &api.IoThrottle{}
			}
			if throttleBW, err := strconv.ParseUint(v, 10, 64); err != nil {
				return nil, nil, nil, err
			} else {
				spec.IoThrottle.WriteBwMbytes = uint32(throttleBW)
			}

		default:
			locator.VolumeLabels[k] = v
		}
	}

	if pureBackendVolName != nil {
		if spec.ProxySpec == nil || !spec.ProxySpec.IsPureBackend() {
			return nil, nil, nil, fmt.Errorf("%s parameter can only be used with Pure DirectAccess volumes", api.SpecBackendVolName)
		}

		if spec.ProxySpec.ProxyProtocol == api.ProxyProtocol_PROXY_PROTOCOL_PURE_BLOCK {
			if spec.ProxySpec.PureBlockSpec == nil {
				spec.ProxySpec.PureBlockSpec = &api.PureBlockSpec{}
			}
			spec.ProxySpec.PureBlockSpec.FullVolName = *pureBackendVolName
		}

		if spec.ProxySpec.ProxyProtocol == api.ProxyProtocol_PROXY_PROTOCOL_PURE_FILE {
			if spec.ProxySpec.PureFileSpec == nil {
				spec.ProxySpec.PureFileSpec = &api.PureFileSpec{}
			}
			spec.ProxySpec.PureFileSpec.FullVolName = *pureBackendVolName
		}
	}

	// Copy any spec labels to the locator
	locator = locator.MergeVolumeSpecLabels(spec)

	return spec, locator, source, nil
}

func (d *specHandler) SpecFromOpts(
	opts map[string]string,
) (*api.VolumeSpec, *api.VolumeLocator, *api.Source, error) {
	source := &api.Source{}
	locator := &api.VolumeLocator{
		VolumeLabels: make(map[string]string),
	}

	spec := d.DefaultSpec()
	return d.UpdateSpecFromOpts(opts, spec, locator, source)
}

func (d *specHandler) GetTokenFromString(str string) (string, bool) {
	ok, token := d.getVal(tokenRegex, str)
	return token, ok
}

func (d *specHandler) GetTokenSecretContextFromString(str string) (*api.TokenSecretContext, bool) {
	submatches := tokenSecretRegex.FindStringSubmatch(str)
	if len(submatches) < 2 {
		// Must at least have a secret name. All other fields are optional,
		// depending on the secrets provider configured.
		return nil, false
	}
	secret := submatches[1]
	secret = strings.TrimRight(secret, "/")

	_, secretNamespace := d.getVal(tokenSecretNamespaceRegex, str)
	return &api.TokenSecretContext{
		SecretName:      secret,
		SecretNamespace: secretNamespace,
	}, true
}

func (d *specHandler) SpecOptsFromString(
	str string,
) (bool, map[string]string, string) {
	// If we can't parse the name, the rest of the spec is invalid.
	ok, name := d.getVal(nameRegex, str)
	if !ok {
		return false, nil, str
	}

	opts := make(map[string]string)

	if ok, sz := d.getVal(sizeRegex, str); ok {
		opts[api.SpecSize] = sz
	}

	if ok, nodes := d.getVal(nodesRegex, str); ok {
		opts[api.SpecNodes] = strings.Replace(nodes, ";", ",", -1)
	}

	if ok, parent := d.getVal(parentRegex, str); ok {
		opts[api.SpecParent] = parent
	}
	if ok, scale := d.getVal(scaleRegex, str); ok {
		opts[api.SpecScale] = scale
	}
	if ok, fs := d.getVal(fsRegex, str); ok {
		opts[api.SpecFilesystem] = fs
	}
	if ok, bs := d.getVal(bsRegex, str); ok {
		opts[api.SpecBlockSize] = bs
	}
	if ok, qd := d.getVal(queueDepthRegex, str); ok {
		opts[api.SpecQueueDepth] = qd
	}
	if ok, ha := d.getVal(haRegex, str); ok {
		opts[api.SpecHaLevel] = ha
	}
	if ok, priority := d.getVal(cosRegex, str); ok {
		opts[api.SpecPriority] = priority
	}
	if ok, shared := d.getVal(sharedRegex, str); ok {
		opts[api.SpecShared] = shared
	}
	if ok, journal := d.getVal(journalRegex, str); ok {
		opts[api.SpecJournal] = journal
	}
	if ok, sharedv4 := d.getVal(sharedv4Regex, str); ok {
		opts[api.SpecSharedv4] = sharedv4
	}
	if ok, cascaded := d.getVal(cascadedRegex, str); ok {
		opts[api.SpecCascaded] = cascaded
	}
	if ok, sticky := d.getVal(stickyRegex, str); ok {
		opts[api.SpecSticky] = sticky
	}
	if ok, secure := d.getVal(secureRegex, str); ok {
		opts[api.SpecSecure] = secure
	}
	if ok, passphrase := d.getVal(passphraseRegex, str); ok {
		opts[api.SpecPassphrase] = passphrase
	}
	if ok, zones := d.getVal(zonesRegex, str); ok {
		opts[api.SpecZones] = zones
	}
	if ok, racks := d.getVal(racksRegex, str); ok {
		opts[api.SpecRacks] = racks
	} else {
		if ok, rack := d.getVal(rackRegex, str); ok {
			opts[api.SpecRack] = rack
		}
	}
	if ok, aggregationLvl := d.getVal(aggrRegex, str); ok {
		opts[api.SpecAggregationLevel] = aggregationLvl
	}
	if ok, compressed := d.getVal(compressedRegex, str); ok {
		opts[api.SpecCompressed] = compressed
	}
	if ok, sched := d.getVal(snapScheduleRegex, str); ok {
		opts[api.SpecSnapshotSchedule] = strings.Replace(sched, "#", ",", -1)
	}
	if ok, ioProfile := d.getVal(ioProfileRegex, str); ok {
		opts[api.SpecIoProfile] = ioProfile
	}
	if ok, asyncIo := d.getVal(asyncIoRegex, str); ok {
		opts[api.SpecAsyncIo] = asyncIo
	}
	if ok, earlyAck := d.getVal(earlyAckRegex, str); ok {
		opts[api.SpecEarlyAck] = earlyAck
	}
	if ok, forceUnsupportedFsType := d.getVal(forceUnsupportedFsTypeRegex, str); ok {
		opts[api.SpecForceUnsupportedFsType] = forceUnsupportedFsType
	}
	if ok, nodiscard := d.getVal(nodiscardRegex, str); ok {
		opts[api.SpecNodiscard] = nodiscard
		// if nodiscard was specified and autofstrim was not specified
		// autfstrim will follow nodoscard
		if ok, _ := d.getVal(AutoFstrimRegex, str); !ok {
			opts[api.SpecAutoFstrim] = nodiscard
		}
	}
	if ok, storagepolicy := d.getVal(storagePolicyRegex, str); ok {
		opts[api.StoragePolicy] = storagepolicy
	}
	if ok, exportProtocol := d.getVal(exportProtocolRegex, str); ok {
		opts[api.SpecExportProtocol] = exportProtocol
	}
	if ok, exportOptions := d.getVal(exportOptionsRegex, str); ok {
		opts[api.SpecExportOptions] = exportOptions
	}
	if ok, proxyEndpoint := d.getVal(proxyEndpointRegex, str); ok {
		opts[api.SpecProxyEndpoint] = proxyEndpoint
	}
	if ok, proxyNFSSubPath := d.getVal(proxyNFSSubPathRegex, str); ok {
		opts[api.SpecProxyNFSSubPath] = proxyNFSSubPath
	}
	if ok, proxyNFSExportPath := d.getVal(proxyNFSExportPathRegex, str); ok {
		opts[api.SpecProxyNFSExportPath] = proxyNFSExportPath
	}
	if ok, proxyS3Bucket := d.getVal(proxyS3BucketRegex, str); ok {
		opts[api.SpecProxyS3Bucket] = proxyS3Bucket
	}
	if ok, mountOptions := d.getVal(mountOptionsRegex, str); ok {
		// mount options will be provided as a string in the following format
		// mount_options=k1;k2:v2;k3:v3
		// convert it to k1,k2=v2,k3=v3
		opts[api.SpecMountOptions] = strings.Replace(strings.Replace(mountOptions, ":", "=", -1), ";", ",", -1)
	}
	if ok, mountOptions := d.getVal(sharedv4MountOptionsRegex, str); ok {
		// mount options will be provided as a string in the following format
		// mount_options=k1;k2:v2;k3:v3
		// convert it to k1,k2=v2,k3=v3
		opts[api.SpecSharedv4MountOptions] = strings.Replace(strings.Replace(mountOptions, ":", "=", -1), ";", ",", -1)
	}

	if ok, cowOnDemand := d.getVal(cowOnDemandRegex, str); ok {
		opts[api.SpecCowOnDemand] = cowOnDemand
	}
	if ok, directIo := d.getVal(directIoRegex, str); ok {
		opts[api.SpecDirectIo] = directIo
	}
	if ok, proxyWrite := d.getVal(ProxyWriteRegex, str); ok {
		opts[api.SpecProxyWrite] = proxyWrite
	}
	if ok, fastpath := d.getVal(fastpathRegex, str); ok {
		opts[api.SpecFastpath] = fastpath
	}
	if ok, sharedv4ServiceType := d.getVal(sharedv4ServiceTypeRegex, str); ok {
		opts[api.SpecSharedv4ServiceType] = sharedv4ServiceType
	}
	if ok, failoverStrategy := d.getVal(Sharedv4FailoverStrategyRegex, str); ok {
		opts[api.SpecSharedv4FailoverStrategy] = failoverStrategy
	}
	if ok, autoFstrim := d.getVal(AutoFstrimRegex, str); ok {
		opts[api.SpecAutoFstrim] = autoFstrim
	}
	if ok, ioThrottleIOPS := d.getVal(SpecIoThrottleRdIOPSRegex, str); ok {
		opts[api.SpecIoThrottleRdIOPS] = ioThrottleIOPS
	}
	if ok, ioThrottleIOPS := d.getVal(SpecIoThrottleWrIOPSRegex, str); ok {
		opts[api.SpecIoThrottleWrIOPS] = ioThrottleIOPS
	}
	if ok, ioThrottleBW := d.getVal(SpecIoThrottleRdBWRegex, str); ok {
		opts[api.SpecIoThrottleRdBW] = ioThrottleBW
	}
	if ok, ioThrottleBW := d.getVal(SpecIoThrottleWrBWRegex, str); ok {
		opts[api.SpecIoThrottleWrBW] = ioThrottleBW
	}
	return true, opts, name
}

func (d *specHandler) SpecFromString(
	str string,
) (bool, *api.VolumeSpec, *api.VolumeLocator, *api.Source, string) {
	ok, opts, name := d.SpecOptsFromString(str)
	if !ok {
		return false, d.DefaultSpec(), nil, nil, name
	}
	spec, locator, source, err := d.SpecFromOpts(opts)
	if err != nil {
		return false, d.DefaultSpec(), nil, nil, name
	}
	return true, spec, locator, source, name
}
