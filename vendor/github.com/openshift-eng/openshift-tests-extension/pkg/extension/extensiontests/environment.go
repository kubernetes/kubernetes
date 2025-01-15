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
