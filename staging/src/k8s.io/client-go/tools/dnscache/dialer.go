package dnscache

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"
)

// Dialer describes a net dialer. This interface is an abstract of net.Dialer.
type Dialer interface {
	// Dial connects to the address on the named network.
	Dial(network, address string) (net.Conn, error)
	// DialContext connects to the address on the named network using the provided context.
	DialContext(ctx context.Context, network, address string) (net.Conn, error)
}

// DialContext implements Dialer.
type DialContext func(ctx context.Context, network, address string) (net.Conn, error)

// Dial connects to the address on the named network.
func (d DialContext) Dial(network, address string) (net.Conn, error) {
	return d(context.Background(), network, address)
}

// DialContext connects to the address on the named network using the provided context.
func (d DialContext) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	return d(ctx, network, address)
}

// Stats returns stats of a dialer.
type Stats interface {
	Stats() DialerStats
}

// Resolver describs a net resolver. This interface is an abstract of net.Resolver.
type Resolver interface {
	LookupIPAddr(ctx context.Context, host string) ([]net.IPAddr, error)
}

// DialerConfig contains configs for creating a dialer with dns cache.
type DialerConfig struct {
	// Dialer is an underlying net dialer.
	Dialer Dialer
	// Resolver is an underlying net resolver.
	Resolver Resolver
	// MinCacheDuration is the minimal cache duration for a dns result.
	MinCacheDuration time.Duration
	// MaxCacheDuration is the maximum cache duration for a dns result.
	// The ttl for a cached dns result is a random value between MinCacheDuration and MaxCacheDuration.
	MaxCacheDuration time.Duration
	// ForceRefreshTimes specifies the max times for a cached dns result to be accessed.
	// If a cached dns result has been accessed ForceRefreshTimes times, this dns result will be
	// forced to deprecate. In this case, MinCacheDuration and MaxCacheDuration are ignored.
	// If this value is less than or equal to 0, it disables dns cache.
	ForceRefreshTimes int64
}

// cacheItem describes a cached dns result.
type cacheItem struct {
	host           string
	ips            []net.IPAddr
	expirationTime time.Time
	usageCount     int64
	maxUsageCount  int64
}

func newCacheItem(host string, ips []net.IPAddr, ttl time.Duration, maxUsageCount int64) *cacheItem {
	return &cacheItem{
		host:           host,
		ips:            ips,
		expirationTime: time.Now().Add(ttl),
		usageCount:     0,
		maxUsageCount:  maxUsageCount,
	}
}

// ip returns an ip and a bool value which indicates whether the cache is valid.
func (i *cacheItem) ip() (net.IPAddr, bool) {
	if len(i.ips) <= 0 {
		return net.IPAddr{}, false
	}
	count := atomic.AddInt64(&i.usageCount, 1)
	// DNS Round-Robin.
	index := int((count - 1)) % len(i.ips)
	return i.ips[index], i.maxUsageCount >= count && time.Now().Before(i.expirationTime)
}

// DialerStats contains stats of a dialer.
type DialerStats struct {
	TotalConn          int64
	CacheMiss          int64
	CacheHit           int64
	DNSQuery           int64
	SuccessfulDNSQuery int64
}

type dialer struct {
	dialer            Dialer
	resolver          Resolver
	minCacheDuration  time.Duration
	maxCacheDuration  time.Duration
	forceRefreshTimes int64
	rand              *rand.Rand

	lock  sync.RWMutex
	cache map[string]*cacheItem

	chanLock        sync.Mutex
	resolveChannels map[string]chan error

	stats DialerStats
}

// NewDialer creates a dialer with dns cache.
func NewDialer(config *DialerConfig) (Dialer, error) {
	d := &dialer{
		dialer:            config.Dialer,
		resolver:          config.Resolver,
		minCacheDuration:  config.MinCacheDuration,
		maxCacheDuration:  config.MaxCacheDuration,
		forceRefreshTimes: config.ForceRefreshTimes,
		rand:              rand.New(rand.NewSource(time.Now().UnixNano())),
		cache:             map[string]*cacheItem{},
		resolveChannels:   map[string]chan error{},
	}
	if d.dialer == nil {
		d.dialer = &net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}
	}
	if d.resolver == nil {
		d.resolver = &net.Resolver{}
	}
	if d.minCacheDuration == 0 {
		d.minCacheDuration = time.Second * 30
	}
	if d.maxCacheDuration == 0 {
		d.maxCacheDuration = time.Minute
	}
	if d.forceRefreshTimes == 0 {
		d.forceRefreshTimes = 10
	}
	if d.forceRefreshTimes < 0 {
		d.forceRefreshTimes = 0
	}
	if d.minCacheDuration > d.maxCacheDuration {
		return nil, fmt.Errorf("min cache duration(%s) should less than or equal to max cache duration(%s)", d.minCacheDuration.String(), d.maxCacheDuration.String())
	}
	return d, nil
}

func (d *dialer) Stats() DialerStats {
	return d.stats
}

func (d *dialer) Dial(network, address string) (net.Conn, error) {
	return d.DialContext(context.Background(), network, address)
}

func (d *dialer) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	atomic.AddInt64(&d.stats.TotalConn, 1)
	if d.forceRefreshTimes > 0 && (network == "tcp" || network == "tcp4") && address != "" {
		host, port, err := net.SplitHostPort(address)
		if err != nil {
			return nil, err
		}
		if host != "" {
			ip, err := d.resolveHost(ctx, host)
			if err == nil {
				atomic.AddInt64(&d.stats.CacheHit, 1)
				// Rewrite address to ip:port.
				address = net.JoinHostPort(ip.String(), port)
			} else {
				atomic.AddInt64(&d.stats.CacheMiss, 1)
				return nil, err
			}
		}
	}
	return d.dialer.DialContext(ctx, network, address)
}

func (d *dialer) resolveHost(ctx context.Context, host string) (net.IPAddr, error) {
	ip, ok := d.getIPFromCache(ctx, host)
	if ok {
		return ip, nil
	}

	d.chanLock.Lock()
	ch := d.resolveChannels[host]
	if ch == nil {
		// Recheck the cache.
		ip, ok := d.getIPFromCache(ctx, host)
		if ok {
			return ip, nil
		}
		ch = make(chan error, 1)
		d.resolveChannels[host] = ch
		// There is no resolving process for the host. Create a new goroutine to lookup dns.
		go func() {
			atomic.AddInt64(&d.stats.DNSQuery, 1)
			var dnsError error
			var item *cacheItem
			defer func() {
				if item != nil {
					atomic.AddInt64(&d.stats.SuccessfulDNSQuery, 1)
				}
				d.lock.Lock()
				if item == nil {
					// Remove host from cache.
					delete(d.cache, host)
				} else {
					// Cache the item.
					d.cache[host] = item
				}
				d.lock.Unlock()

				// Wake up a resolveHost function.
				d.chanLock.Lock()
				delete(d.resolveChannels, host)
				ch <- dnsError
				d.chanLock.Unlock()
			}()
			// Resolve host.
			ips, err := d.resolver.LookupIPAddr(ctx, host)
			if err != nil {
				glog.V(2).Infof("Can't resolve host %s because %s", host, err.Error())
				dnsError = err
				return
			}
			if len(ips) <= 0 {
				glog.V(2).Infof("Can't resolve host %s because the host has no ip", host)
				dnsError = fmt.Errorf("no dns records for host %s", host)
				return
			}
			item = newCacheItem(host, ips, d.randomTTL(), d.forceRefreshTimes)
			return
		}()
	}
	d.chanLock.Unlock()

	select {
	case err := <-ch:
		// Put back the error. This operation can wake up other resolveHost functions.
		ch <- err
		if err != nil {
			return net.IPAddr{}, err
		}
		ip, _ := d.getIPFromCache(ctx, host)
		// In this case, the dns result is fresh and we can ignore the second result safely.
		if ip.IP != nil {
			return ip, nil
		}
		return net.IPAddr{}, fmt.Errorf("no dns records of host %s in cache", host)
	case <-ctx.Done():
		glog.V(2).Infof("Can't resolve host %s because context timeout", host)
		return net.IPAddr{}, ctx.Err()
	}
}

func (d *dialer) getIPFromCache(ctx context.Context, host string) (net.IPAddr, bool) {
	d.lock.RLock()
	item, ok := d.cache[host]
	d.lock.RUnlock()
	if ok {
		return item.ip()
	}
	return net.IPAddr{}, false
}

func (d *dialer) randomTTL() time.Duration {
	if d.maxCacheDuration == d.minCacheDuration {
		return d.minCacheDuration
	}
	ttl := d.rand.Int63n(int64(d.maxCacheDuration - d.minCacheDuration))
	return d.minCacheDuration + time.Duration(ttl)
}
