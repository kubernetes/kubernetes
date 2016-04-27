package io.k8s.cassandra;

import com.google.common.collect.ImmutableMap;
import org.apache.cassandra.locator.SeedProvider;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.hamcrest.Matchers.*;

import java.net.InetAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class KubernetesSeedProviderTest {

    private static final Logger logger = LoggerFactory.getLogger(KubernetesSeedProviderTest.class);

    @Test
    @Ignore("has to be run inside of a kube cluster")
    public void getSeeds() throws Exception {
        SeedProvider provider = new KubernetesSeedProvider(new HashMap<String, String>());
        List<InetAddress> seeds = provider.getSeeds();

        assertThat(seeds, is(not(empty())));

    }

    @Test
    public void testDefaultSeeds() throws  Exception {
       Map<String, String> params = ImmutableMap.of(
            "seeds", "192.168.1.0,8.8.8.8"
       );

        KubernetesSeedProvider provider = new KubernetesSeedProvider(params);
        List<InetAddress>  seeds = provider.getDefaultSeeds();
        List<InetAddress> seedsTest = new ArrayList<>();
        seedsTest.add(InetAddress.getByName("8.4.4.4"));
        seedsTest.add(InetAddress.getByName("8.8.8.8"));
        assertThat(seeds, is(not(empty())));
        assertThat(seeds, is(seedsTest));
        logger.debug("seeds loaded {}", seeds);

    }


}