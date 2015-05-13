package io.k8s.cassandra;

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.net.URL;
import java.net.URLConnection;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.codehaus.jackson.JsonNode;
import org.codehaus.jackson.annotate.JsonIgnoreProperties;
import org.codehaus.jackson.map.ObjectMapper;
import org.apache.cassandra.locator.SeedProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KubernetesSeedProvider implements SeedProvider {
    
    @JsonIgnoreProperties(ignoreUnknown = true)
    static class Address {
        public String IP;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static class Subset {
        public List<Address> addresses; 
    }
    
    @JsonIgnoreProperties(ignoreUnknown = true)
    static class Endpoints {
        public List<Subset> subsets;
    }
    
    private static String getEnvOrDefault(String var, String def) {
        String val = System.getenv(var);
        if (val == null) {
	    val = def;
        }
        return val;
    }

    private static final Logger logger = LoggerFactory.getLogger(KubernetesSeedProvider.class);

    private List defaultSeeds;
   
    public KubernetesSeedProvider(Map<String, String> params) {
        // Taken from SimpleSeedProvider.java
        // These are used as a fallback, if we get nothing from k8s.
        String[] hosts = params.get("seeds").split(",", -1);
        defaultSeeds = new ArrayList<InetAddress>(hosts.length);
        for (String host : hosts)
	    {
		try {
		    defaultSeeds.add(InetAddress.getByName(host.trim()));
		}
		catch (UnknownHostException ex)
		    {
			// not fatal... DD will bark if there end up being zero seeds.
			logger.warn("Seed provider couldn't lookup host " + host);
		    }
	    }
    } 

    public List<InetAddress> getSeeds() {
        List<InetAddress> list = new ArrayList<InetAddress>();
        String protocol = getEnvOrDefault("KUBERNETES_API_PROTOCOL", "http");
        String hostName = getEnvOrDefault("KUBERNETES_RO_SERVICE_HOST", "localhost");
        String hostPort = getEnvOrDefault("KUBERNETES_RO_SERVICE_PORT", "8080");

        String host = protocol + "://" + hostName + ":" + hostPort;
        String serviceName = getEnvOrDefault("CASSANDRA_SERVICE", "cassandra");
        String path = "/api/v1beta3/namespaces/default/endpoints/";
        try {
            URL url = new URL(host + path + serviceName);
            ObjectMapper mapper = new ObjectMapper();
            Endpoints endpoints = mapper.readValue(url, Endpoints.class);
            if (endpoints != null) {
                // Here is a problem point, endpoints.subsets can be null in first node cases.
                if (endpoints.subsets != null && !endpoints.subsets.isEmpty()){
                    for (Subset subset : endpoints.subsets) {
                        for (Address address : subset.addresses) {
                            list.add(InetAddress.getByName(address.IP));
                        }
                    }
                }
            }
        } catch (IOException ex) {
	    logger.warn("Request to kubernetes apiserver failed"); 
        }
        if (list.size() == 0) {
	    // If we got nothing, we might be the first instance, in that case
	    // fall back on the seeds that were passed in cassandra.yaml.
	    return defaultSeeds;
        }
        return list;
    }

    // Simple main to test the implementation
    public static void main(String[] args) {
        SeedProvider provider = new KubernetesSeedProvider(new HashMap<String, String>());
        System.out.println(provider.getSeeds());
    }
}
