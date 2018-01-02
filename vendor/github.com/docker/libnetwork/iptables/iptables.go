package iptables

import (
	"errors"
	"fmt"
	"net"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/sirupsen/logrus"
)

// Action signifies the iptable action.
type Action string

// Policy is the default iptable policies
type Policy string

// Table refers to Nat, Filter or Mangle.
type Table string

const (
	// Append appends the rule at the end of the chain.
	Append Action = "-A"
	// Delete deletes the rule from the chain.
	Delete Action = "-D"
	// Insert inserts the rule at the top of the chain.
	Insert Action = "-I"
	// Nat table is used for nat translation rules.
	Nat Table = "nat"
	// Filter table is used for filter rules.
	Filter Table = "filter"
	// Mangle table is used for mangling the packet.
	Mangle Table = "mangle"
	// Drop is the default iptables DROP policy
	Drop Policy = "DROP"
	// Accept is the default iptables ACCEPT policy
	Accept Policy = "ACCEPT"
)

var (
	iptablesPath  string
	supportsXlock = false
	supportsCOpt  = false
	xLockWaitMsg  = "Another app is currently holding the xtables lock; waiting"
	// used to lock iptables commands if xtables lock is not supported
	bestEffortLock sync.Mutex
	// ErrIptablesNotFound is returned when the rule is not found.
	ErrIptablesNotFound = errors.New("Iptables not found")
	initOnce            sync.Once
)

// ChainInfo defines the iptables chain.
type ChainInfo struct {
	Name        string
	Table       Table
	HairpinMode bool
}

// ChainError is returned to represent errors during ip table operation.
type ChainError struct {
	Chain  string
	Output []byte
}

func (e ChainError) Error() string {
	return fmt.Sprintf("Error iptables %s: %s", e.Chain, string(e.Output))
}

func probe() {
	if out, err := exec.Command("modprobe", "-va", "nf_nat").CombinedOutput(); err != nil {
		logrus.Warnf("Running modprobe nf_nat failed with message: `%s`, error: %v", strings.TrimSpace(string(out)), err)
	}
	if out, err := exec.Command("modprobe", "-va", "xt_conntrack").CombinedOutput(); err != nil {
		logrus.Warnf("Running modprobe xt_conntrack failed with message: `%s`, error: %v", strings.TrimSpace(string(out)), err)
	}
}

func initFirewalld() {
	if err := FirewalldInit(); err != nil {
		logrus.Debugf("Fail to initialize firewalld: %v, using raw iptables instead", err)
	}
}

func detectIptables() {
	path, err := exec.LookPath("iptables")
	if err != nil {
		return
	}
	iptablesPath = path
	supportsXlock = exec.Command(iptablesPath, "--wait", "-L", "-n").Run() == nil
	mj, mn, mc, err := GetVersion()
	if err != nil {
		logrus.Warnf("Failed to read iptables version: %v", err)
		return
	}
	supportsCOpt = supportsCOption(mj, mn, mc)
}

func initDependencies() {
	probe()
	initFirewalld()
	detectIptables()
}

func initCheck() error {
	initOnce.Do(initDependencies)

	if iptablesPath == "" {
		return ErrIptablesNotFound
	}
	return nil
}

// NewChain adds a new chain to ip table.
func NewChain(name string, table Table, hairpinMode bool) (*ChainInfo, error) {
	c := &ChainInfo{
		Name:        name,
		Table:       table,
		HairpinMode: hairpinMode,
	}
	if string(c.Table) == "" {
		c.Table = Filter
	}

	// Add chain if it doesn't exist
	if _, err := Raw("-t", string(c.Table), "-n", "-L", c.Name); err != nil {
		if output, err := Raw("-t", string(c.Table), "-N", c.Name); err != nil {
			return nil, err
		} else if len(output) != 0 {
			return nil, fmt.Errorf("Could not create %s/%s chain: %s", c.Table, c.Name, output)
		}
	}
	return c, nil
}

// ProgramChain is used to add rules to a chain
func ProgramChain(c *ChainInfo, bridgeName string, hairpinMode, enable bool) error {
	if c.Name == "" {
		return errors.New("Could not program chain, missing chain name")
	}

	switch c.Table {
	case Nat:
		preroute := []string{
			"-m", "addrtype",
			"--dst-type", "LOCAL",
			"-j", c.Name}
		if !Exists(Nat, "PREROUTING", preroute...) && enable {
			if err := c.Prerouting(Append, preroute...); err != nil {
				return fmt.Errorf("Failed to inject %s in PREROUTING chain: %s", c.Name, err)
			}
		} else if Exists(Nat, "PREROUTING", preroute...) && !enable {
			if err := c.Prerouting(Delete, preroute...); err != nil {
				return fmt.Errorf("Failed to remove %s in PREROUTING chain: %s", c.Name, err)
			}
		}
		output := []string{
			"-m", "addrtype",
			"--dst-type", "LOCAL",
			"-j", c.Name}
		if !hairpinMode {
			output = append(output, "!", "--dst", "127.0.0.0/8")
		}
		if !Exists(Nat, "OUTPUT", output...) && enable {
			if err := c.Output(Append, output...); err != nil {
				return fmt.Errorf("Failed to inject %s in OUTPUT chain: %s", c.Name, err)
			}
		} else if Exists(Nat, "OUTPUT", output...) && !enable {
			if err := c.Output(Delete, output...); err != nil {
				return fmt.Errorf("Failed to inject %s in OUTPUT chain: %s", c.Name, err)
			}
		}
	case Filter:
		if bridgeName == "" {
			return fmt.Errorf("Could not program chain %s/%s, missing bridge name",
				c.Table, c.Name)
		}
		link := []string{
			"-o", bridgeName,
			"-j", c.Name}
		if !Exists(Filter, "FORWARD", link...) && enable {
			insert := append([]string{string(Insert), "FORWARD"}, link...)
			if output, err := Raw(insert...); err != nil {
				return err
			} else if len(output) != 0 {
				return fmt.Errorf("Could not create linking rule to %s/%s: %s", c.Table, c.Name, output)
			}
		} else if Exists(Filter, "FORWARD", link...) && !enable {
			del := append([]string{string(Delete), "FORWARD"}, link...)
			if output, err := Raw(del...); err != nil {
				return err
			} else if len(output) != 0 {
				return fmt.Errorf("Could not delete linking rule from %s/%s: %s", c.Table, c.Name, output)
			}

		}
		establish := []string{
			"-o", bridgeName,
			"-m", "conntrack",
			"--ctstate", "RELATED,ESTABLISHED",
			"-j", "ACCEPT"}
		if !Exists(Filter, "FORWARD", establish...) && enable {
			insert := append([]string{string(Insert), "FORWARD"}, establish...)
			if output, err := Raw(insert...); err != nil {
				return err
			} else if len(output) != 0 {
				return fmt.Errorf("Could not create establish rule to %s: %s", c.Table, output)
			}
		} else if Exists(Filter, "FORWARD", establish...) && !enable {
			del := append([]string{string(Delete), "FORWARD"}, establish...)
			if output, err := Raw(del...); err != nil {
				return err
			} else if len(output) != 0 {
				return fmt.Errorf("Could not delete establish rule from %s: %s", c.Table, output)
			}
		}
	}
	return nil
}

// RemoveExistingChain removes existing chain from the table.
func RemoveExistingChain(name string, table Table) error {
	c := &ChainInfo{
		Name:  name,
		Table: table,
	}
	if string(c.Table) == "" {
		c.Table = Filter
	}
	return c.Remove()
}

// Forward adds forwarding rule to 'filter' table and corresponding nat rule to 'nat' table.
func (c *ChainInfo) Forward(action Action, ip net.IP, port int, proto, destAddr string, destPort int, bridgeName string) error {
	daddr := ip.String()
	if ip.IsUnspecified() {
		// iptables interprets "0.0.0.0" as "0.0.0.0/32", whereas we
		// want "0.0.0.0/0". "0/0" is correctly interpreted as "any
		// value" by both iptables and ip6tables.
		daddr = "0/0"
	}

	args := []string{
		"-p", proto,
		"-d", daddr,
		"--dport", strconv.Itoa(port),
		"-j", "DNAT",
		"--to-destination", net.JoinHostPort(destAddr, strconv.Itoa(destPort))}
	if !c.HairpinMode {
		args = append(args, "!", "-i", bridgeName)
	}
	if err := ProgramRule(Nat, c.Name, action, args); err != nil {
		return err
	}

	args = []string{
		"!", "-i", bridgeName,
		"-o", bridgeName,
		"-p", proto,
		"-d", destAddr,
		"--dport", strconv.Itoa(destPort),
		"-j", "ACCEPT",
	}
	if err := ProgramRule(Filter, c.Name, action, args); err != nil {
		return err
	}

	args = []string{
		"-p", proto,
		"-s", destAddr,
		"-d", destAddr,
		"--dport", strconv.Itoa(destPort),
		"-j", "MASQUERADE",
	}
	if err := ProgramRule(Nat, "POSTROUTING", action, args); err != nil {
		return err
	}

	return nil
}

// Link adds reciprocal ACCEPT rule for two supplied IP addresses.
// Traffic is allowed from ip1 to ip2 and vice-versa
func (c *ChainInfo) Link(action Action, ip1, ip2 net.IP, port int, proto string, bridgeName string) error {
	// forward
	args := []string{
		"-i", bridgeName, "-o", bridgeName,
		"-p", proto,
		"-s", ip1.String(),
		"-d", ip2.String(),
		"--dport", strconv.Itoa(port),
		"-j", "ACCEPT",
	}
	if err := ProgramRule(Filter, c.Name, action, args); err != nil {
		return err
	}
	// reverse
	args[7], args[9] = args[9], args[7]
	args[10] = "--sport"
	if err := ProgramRule(Filter, c.Name, action, args); err != nil {
		return err
	}
	return nil
}

// ProgramRule adds the rule specified by args only if the
// rule is not already present in the chain. Reciprocally,
// it removes the rule only if present.
func ProgramRule(table Table, chain string, action Action, args []string) error {
	if Exists(table, chain, args...) != (action == Delete) {
		return nil
	}
	return RawCombinedOutput(append([]string{"-t", string(table), string(action), chain}, args...)...)
}

// Prerouting adds linking rule to nat/PREROUTING chain.
func (c *ChainInfo) Prerouting(action Action, args ...string) error {
	a := []string{"-t", string(Nat), string(action), "PREROUTING"}
	if len(args) > 0 {
		a = append(a, args...)
	}
	if output, err := Raw(a...); err != nil {
		return err
	} else if len(output) != 0 {
		return ChainError{Chain: "PREROUTING", Output: output}
	}
	return nil
}

// Output adds linking rule to an OUTPUT chain.
func (c *ChainInfo) Output(action Action, args ...string) error {
	a := []string{"-t", string(c.Table), string(action), "OUTPUT"}
	if len(args) > 0 {
		a = append(a, args...)
	}
	if output, err := Raw(a...); err != nil {
		return err
	} else if len(output) != 0 {
		return ChainError{Chain: "OUTPUT", Output: output}
	}
	return nil
}

// Remove removes the chain.
func (c *ChainInfo) Remove() error {
	// Ignore errors - This could mean the chains were never set up
	if c.Table == Nat {
		c.Prerouting(Delete, "-m", "addrtype", "--dst-type", "LOCAL", "-j", c.Name)
		c.Output(Delete, "-m", "addrtype", "--dst-type", "LOCAL", "!", "--dst", "127.0.0.0/8", "-j", c.Name)
		c.Output(Delete, "-m", "addrtype", "--dst-type", "LOCAL", "-j", c.Name) // Created in versions <= 0.1.6

		c.Prerouting(Delete)
		c.Output(Delete)
	}
	Raw("-t", string(c.Table), "-F", c.Name)
	Raw("-t", string(c.Table), "-X", c.Name)
	return nil
}

// Exists checks if a rule exists
func Exists(table Table, chain string, rule ...string) bool {
	return exists(false, table, chain, rule...)
}

// ExistsNative behaves as Exists with the difference it
// will always invoke `iptables` binary.
func ExistsNative(table Table, chain string, rule ...string) bool {
	return exists(true, table, chain, rule...)
}

func exists(native bool, table Table, chain string, rule ...string) bool {
	f := Raw
	if native {
		f = raw
	}

	if string(table) == "" {
		table = Filter
	}

	if err := initCheck(); err != nil {
		// The exists() signature does not allow us to return an error, but at least
		// we can skip the (likely invalid) exec invocation.
		return false
	}

	if supportsCOpt {
		// if exit status is 0 then return true, the rule exists
		_, err := f(append([]string{"-t", string(table), "-C", chain}, rule...)...)
		return err == nil
	}

	// parse "iptables -S" for the rule (it checks rules in a specific chain
	// in a specific table and it is very unreliable)
	return existsRaw(table, chain, rule...)
}

func existsRaw(table Table, chain string, rule ...string) bool {
	ruleString := fmt.Sprintf("%s %s\n", chain, strings.Join(rule, " "))
	existingRules, _ := exec.Command(iptablesPath, "-t", string(table), "-S", chain).Output()

	return strings.Contains(string(existingRules), ruleString)
}

// Raw calls 'iptables' system command, passing supplied arguments.
func Raw(args ...string) ([]byte, error) {
	if firewalldRunning {
		output, err := Passthrough(Iptables, args...)
		if err == nil || !strings.Contains(err.Error(), "was not provided by any .service files") {
			return output, err
		}
	}
	return raw(args...)
}

func raw(args ...string) ([]byte, error) {
	if err := initCheck(); err != nil {
		return nil, err
	}
	if supportsXlock {
		args = append([]string{"--wait"}, args...)
	} else {
		bestEffortLock.Lock()
		defer bestEffortLock.Unlock()
	}

	logrus.Debugf("%s, %v", iptablesPath, args)

	output, err := exec.Command(iptablesPath, args...).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("iptables failed: iptables %v: %s (%s)", strings.Join(args, " "), output, err)
	}

	// ignore iptables' message about xtables lock
	if strings.Contains(string(output), xLockWaitMsg) {
		output = []byte("")
	}

	return output, err
}

// RawCombinedOutput inernally calls the Raw function and returns a non nil
// error if Raw returned a non nil error or a non empty output
func RawCombinedOutput(args ...string) error {
	if output, err := Raw(args...); err != nil || len(output) != 0 {
		return fmt.Errorf("%s (%v)", string(output), err)
	}
	return nil
}

// RawCombinedOutputNative behave as RawCombinedOutput with the difference it
// will always invoke `iptables` binary
func RawCombinedOutputNative(args ...string) error {
	if output, err := raw(args...); err != nil || len(output) != 0 {
		return fmt.Errorf("%s (%v)", string(output), err)
	}
	return nil
}

// ExistChain checks if a chain exists
func ExistChain(chain string, table Table) bool {
	if _, err := Raw("-t", string(table), "-L", chain); err == nil {
		return true
	}
	return false
}

// GetVersion reads the iptables version numbers during initialization
func GetVersion() (major, minor, micro int, err error) {
	out, err := exec.Command(iptablesPath, "--version").CombinedOutput()
	if err == nil {
		major, minor, micro = parseVersionNumbers(string(out))
	}
	return
}

// SetDefaultPolicy sets the passed default policy for the table/chain
func SetDefaultPolicy(table Table, chain string, policy Policy) error {
	if err := RawCombinedOutput("-t", string(table), "-P", chain, string(policy)); err != nil {
		return fmt.Errorf("setting default policy to %v in %v chain failed: %v", policy, chain, err)
	}
	return nil
}

func parseVersionNumbers(input string) (major, minor, micro int) {
	re := regexp.MustCompile(`v\d*.\d*.\d*`)
	line := re.FindString(input)
	fmt.Sscanf(line, "v%d.%d.%d", &major, &minor, &micro)
	return
}

// iptables -C, --check option was added in v.1.4.11
// http://ftp.netfilter.org/pub/iptables/changes-iptables-1.4.11.txt
func supportsCOption(mj, mn, mc int) bool {
	return mj > 1 || (mj == 1 && (mn > 4 || (mn == 4 && mc >= 11)))
}

// AddReturnRule adds a return rule for the chain in the filter table
func AddReturnRule(chain string) error {
	var (
		table = Filter
		args  = []string{"-j", "RETURN"}
	)

	if Exists(table, chain, args...) {
		return nil
	}

	err := RawCombinedOutput(append([]string{"-A", chain}, args...)...)
	if err != nil {
		return fmt.Errorf("unable to add return rule in %s chain: %s", chain, err.Error())
	}

	return nil
}

// EnsureJumpRule ensures the jump rule is on top
func EnsureJumpRule(fromChain, toChain string) error {
	var (
		table = Filter
		args  = []string{"-j", toChain}
	)

	if Exists(table, fromChain, args...) {
		err := RawCombinedOutput(append([]string{"-D", fromChain}, args...)...)
		if err != nil {
			return fmt.Errorf("unable to remove jump to %s rule in %s chain: %s", toChain, fromChain, err.Error())
		}
	}

	err := RawCombinedOutput(append([]string{"-I", fromChain}, args...)...)
	if err != nil {
		return fmt.Errorf("unable to insert jump to %s rule in %s chain: %s", toChain, fromChain, err.Error())
	}

	return nil
}
