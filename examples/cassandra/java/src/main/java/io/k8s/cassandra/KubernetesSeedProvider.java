/*
 * Copyright (C) 2015 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package io.k8s.cassandra;

import org.apache.cassandra.config.Config;
import org.apache.cassandra.config.ConfigurationLoader;
import org.apache.cassandra.config.YamlConfigurationLoader;
import org.apache.cassandra.exceptions.ConfigurationException;
import org.apache.cassandra.locator.SeedProvider;
import org.apache.cassandra.locator.SimpleSeedProvider;
import org.apache.cassandra.utils.FBUtilities;
import org.codehaus.jackson.annotate.JsonIgnoreProperties;
import org.codehaus.jackson.map.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.*;
import java.io.IOException;
import java.net.InetAddress;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Self discovery {@link SeedProvider} that creates a list of Cassandra Seeds by
 * communicating with the Kubernetes API.
 * <p>Various System Variable can be used to configure this provider:
 * <ul>
 *     <li>KUBERNETES_PORT_443_TCP_ADDR defaults to kubernetes.default.svc.cluster.local</li>
 *     <li>KUBERNETES_PORT_443_TCP_PORT defaults to 443</li>
 *     <li>CASSANDRA_SERVICE defaults to cassandra</li>
 *     <li>POD_NAMESPACE defaults to 'default'</li>
 *     <li>CASSANDRA_SERVICE_NUM_SEEDS defaults to 8 seeds</li>
 * </ul>
 */
public class KubernetesSeedProvider implements SeedProvider {

    private static final Logger logger = LoggerFactory.getLogger(KubernetesSeedProvider.class);

    /**
     * default seeds to fall back on
     */
    private List<InetAddress> defaultSeeds;

    private TrustManager[] trustAll;

    private HostnameVerifier trustAllHosts;

    /**
     * Create new Seeds
     * @param params
     */
    public KubernetesSeedProvider(Map<String, String> params) {

        // Create default seeds
        defaultSeeds = createDefaultSeeds();

        // TODO: Load the CA cert when it is available on all platforms.
        trustAll = new TrustManager[] {
            new X509TrustManager() {
                public void checkServerTrusted(X509Certificate[] certs, String authType) {}
                public void checkClientTrusted(X509Certificate[] certs, String authType) {}
                public X509Certificate[] getAcceptedIssuers() { return null; }
            }
        };

        trustAllHosts = new HostnameVerifier() {
            public boolean verify(String hostname, SSLSession session) {
                return true;
            }
        };
    }

    /**
     * Call kubernetes API to collect a list of seed providers
     * @return list of seed providers
     */
    public List<InetAddress> getSeeds() {

        String host = getEnvOrDefault("KUBERNETES_PORT_443_TCP_ADDR", "kubernetes.default.svc.cluster.local");
        String port = getEnvOrDefault("KUBERNETES_PORT_443_TCP_PORT", "443");
        String serviceName = getEnvOrDefault("CASSANDRA_SERVICE", "cassandra");
        String podNamespace = getEnvOrDefault("POD_NAMESPACE", "default");
        String path = String.format("/api/v1/namespaces/%s/endpoints/", podNamespace);
        String seedSizeVar = getEnvOrDefault("CASSANDRA_SERVICE_NUM_SEEDS", "8");
        Integer seedSize = Integer.valueOf(seedSizeVar);

        List<InetAddress> seeds = new ArrayList<InetAddress>();
        try {
            String token = getServiceAccountToken();

            SSLContext ctx = SSLContext.getInstance("SSL");
            ctx.init(null, trustAll, new SecureRandom());

            String PROTO = "https://";
            URL url = new URL(PROTO + host + ":" + port + path + serviceName);
            logger.info("Getting endpoints from " + url);
            HttpsURLConnection conn = (HttpsURLConnection)url.openConnection();

            // TODO: Remove this once the CA cert is propagated everywhere, and replace
            // with loading the CA cert.
            conn.setHostnameVerifier(trustAllHosts);

            conn.setSSLSocketFactory(ctx.getSocketFactory());
            conn.addRequestProperty("Authorization", "Bearer " + token);
            ObjectMapper mapper = new ObjectMapper();
            Endpoints endpoints = mapper.readValue(conn.getInputStream(), Endpoints.class);

            if (endpoints != null) {
                // Here is a problem point, endpoints.subsets can be null in first node cases.
                if (endpoints.subsets != null && !endpoints.subsets.isEmpty()){
                    for (Subset subset : endpoints.subsets) {
                        if (subset.addresses != null && !subset.addresses.isEmpty()) {
                            for (Address address : subset.addresses) {
                                seeds.add(InetAddress.getByName(address.ip));

                                if(seeds.size() >= seedSize) {
                                    logger.info("Available num endpoints: " + seeds.size());
                                    return Collections.unmodifiableList(seeds);
                                }
                            }
                        }
                    }
                }
                logger.info("Available num endpoints: " + seeds.size());
            } else {
                logger.warn("Endpoints are not available using default seeds in cassandra.yaml");
                return Collections.unmodifiableList(defaultSeeds);
            }
        } catch (IOException | NoSuchAlgorithmException | KeyManagementException ex) {
            logger.warn("Request to kubernetes apiserver failed, using default seeds in cassandra.yaml", ex);
            return Collections.unmodifiableList(defaultSeeds);
        }

        if (seeds.size() == 0) {
            // If we got nothing, we might be the first instance, in that case
            // fall back on the seeds that were passed in cassandra.yaml.
            logger.warn("Seeds are not available using default seeds in cassandra.yaml");
            return Collections.unmodifiableList(defaultSeeds);
        }

        return Collections.unmodifiableList(seeds);
    }

    /**
     * Code taken from {@link SimpleSeedProvider}.  This is used as a fall back
     * incase we don't find seeds
     * @return
     */
    protected List<InetAddress> createDefaultSeeds()
    {
        Config conf;
        try {
            conf = loadConfig();
        }
        catch (Exception e) {
            throw new AssertionError(e);
        }
        String[] hosts = conf.seed_provider.parameters.get("seeds").split(",", -1);
        List<InetAddress> seeds = new ArrayList<InetAddress>();
        for (String host : hosts) {
            try {
                seeds.add(InetAddress.getByName(host.trim()));
            }
            catch (UnknownHostException ex) {
                // not fatal... DD will bark if there end up being zero seeds.
                logger.warn("Seed provider couldn't lookup host {}", host);
            }
        }

        if(seeds.size() == 0) {
            try {
                seeds.add(InetAddress.getLocalHost());
            } catch (UnknownHostException e) {
                logger.warn("Seed provider couldn't lookup localhost");
            }
        }
        return Collections.unmodifiableList(seeds);
    }

    /**
     * Code taken from {@link SimpleSeedProvider}
     * @return
     */
    protected static Config loadConfig() throws ConfigurationException
    {
        String loaderClass = System.getProperty("cassandra.config.loader");
        ConfigurationLoader loader = loaderClass == null
                ? new YamlConfigurationLoader()
                : FBUtilities.<ConfigurationLoader>construct(loaderClass, "configuration loading");
        return loader.loadConfig();
    }

    private static String getEnvOrDefault(String var, String def) {
        String val = System.getenv(var);
        if (val == null) {
            val = def;
        }
        return val;
    }

    private static String getServiceAccountToken()  throws IOException {
        String file = "/var/run/secrets/kubernetes.io/serviceaccount/token";
        try {
            return new String(Files.readAllBytes(Paths.get(file)));
        } catch (IOException e) {
            logger.warn("unable to load service account token");
            throw e;
        }
    }

    protected List<InetAddress> getDefaultSeeds() {
        return defaultSeeds;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static class Address {
        public String ip;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static class Subset {
        public List<Address> addresses;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static class Endpoints {
        public List<Subset> subsets;
    }
}
