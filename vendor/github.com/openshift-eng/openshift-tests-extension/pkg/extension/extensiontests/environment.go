package extensiontests

import (
	"fmt"
	"strings"
)

func PlatformEquals(platform string) string {
	return fmt.Sprintf(`platform=="%s"`, platform)
}

func NetworkEquals(network string) string {
	return fmt.Sprintf(`network=="%s"`, network)
}

func NetworkStackEquals(networkStack string) string {
	return fmt.Sprintf(`networkStack=="%s"`, networkStack)
}

func UpgradeEquals(upgrade string) string {
	return fmt.Sprintf(`upgrade=="%s"`, upgrade)
}

func TopologyEquals(topology string) string {
	return fmt.Sprintf(`topology=="%s"`, topology)
}

func ArchitectureEquals(arch string) string {
	return fmt.Sprintf(`architecture=="%s"`, arch)
}

func APIGroupEnabled(apiGroup string) string {
	return fmt.Sprintf(`apiGroups.exists(api, api=="%s")`, apiGroup)
}

func APIGroupDisabled(apiGroup string) string {
	return fmt.Sprintf(`!apiGroups.exists(api, api=="%s")`, apiGroup)
}

func FeatureGateEnabled(featureGate string) string {
	return fmt.Sprintf(`featureGates.exists(fg, fg=="%s")`, featureGate)
}

func FeatureGateDisabled(featureGate string) string {
	return fmt.Sprintf(`!featureGates.exists(fg, fg=="%s")`, featureGate)
}

func ExternalConnectivityEquals(externalConnectivity string) string {
	return fmt.Sprintf(`externalConnectivity=="%s"`, externalConnectivity)
}

func OptionalCapabilitiesIncludeAny(optionalCapability ...string) string {
	for i := range optionalCapability {
		optionalCapability[i] = OptionalCapabilityExists(optionalCapability[i])
	}
	return fmt.Sprintf("(%s)", fmt.Sprint(strings.Join(optionalCapability, " || ")))
}

func OptionalCapabilitiesIncludeAll(optionalCapability ...string) string {
	for i := range optionalCapability {
		optionalCapability[i] = OptionalCapabilityExists(optionalCapability[i])
	}
	return fmt.Sprintf("(%s)", fmt.Sprint(strings.Join(optionalCapability, " && ")))
}

func OptionalCapabilityExists(optionalCapability string) string {
	return fmt.Sprintf(`optionalCapabilities.exists(oc, oc=="%s")`, optionalCapability)
}

func NoOptionalCapabilitiesExist() string {
	return "size(optionalCapabilities) == 0"
}

func InstallerEquals(installer string) string {
	return fmt.Sprintf(`installer=="%s"`, installer)
}

func VersionEquals(version string) string {
	return fmt.Sprintf(`version=="%s"`, version)
}

func FactEquals(key, value string) string {
	return fmt.Sprintf(`(fact_keys.exists(k, k=="%s") && facts["%s"].matches("%s"))`, key, key, value)
}

func Or(cel ...string) string {
	return fmt.Sprintf("(%s)", strings.Join(cel, " || "))
}

func And(cel ...string) string {
	return fmt.Sprintf("(%s)", strings.Join(cel, " && "))
}
