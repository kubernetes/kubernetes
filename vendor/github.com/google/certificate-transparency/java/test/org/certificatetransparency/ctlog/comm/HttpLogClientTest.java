package org.certificatetransparency.ctlog.comm;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.codec.binary.Base64;
import org.apache.http.NameValuePair;
import org.apache.http.message.BasicNameValuePair;
import org.certificatetransparency.ctlog.CertificateTransparencyException;
import org.certificatetransparency.ctlog.ParsedLogEntry;
import org.certificatetransparency.ctlog.ParsedLogEntryWithProof;
import org.certificatetransparency.ctlog.TestData;
import org.certificatetransparency.ctlog.proto.Ct;
import org.certificatetransparency.ctlog.serialization.CryptoDataLoader;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Matchers;

import com.google.protobuf.ByteString;


/**
 * Test interaction with the Log http server.
 */
@RunWith(JUnit4.class)
public class HttpLogClientTest {
  public static final String TEST_DATA_PATH = "test/testdata/test-colliding-roots.pem";

  public static final String STH_RESPONSE = ""
      + "{\"timestamp\":1402415255382,"
      + "\"tree_head_signature\":\"BAMARzBFAiBX9fHXbK3Yi+P+bGM8mlL8XFmwZ7fkbhK2GqlnoJkMkQIhANGoUuD+"
      + "JvjFTRdESfKO5428e1HAQL412Sa5e16D4E3M\","
      + "\"sha256_root_hash\":\"jdH9k+\\/lb9abMz3N8rVmwrw8MWU7v55+nSAXej3hqPg=\","
      + "\"tree_size\":4301837}";
  
  public static final String BAD_STH_RESPONSE_INVALID_TIMESTAMP = ""
      + "{\"timestamp\":-1,"
      + "\"tree_head_signature\":\"BAMARzBFAiBX9fHXbK3Yi+P+bGM8mlL8XFmwZ7fkbhK2GqlnoJkMkQIhANGoUuD+"
      + "JvjFTRdESfKO5428e1HAQL412Sa5e16D4E3M\","
      + "\"sha256_root_hash\":\"jdH9k+\\/lb9abMz3N8rVmwrw8MWU7v55+nSAXej3hqPg=\","
      + "\"tree_size\":0}";
  
  public static final String BAD_STH_RESPONSE_INVALID_ROOT_HASH = ""
          + "{\"timestamp\":1402415255382,"
          + "\"tree_head_signature\":\"BAMARzBFAiBX9fHXbK3Yi+P+bGM8mlL8XFmwZ7fkbhK2GqlnoJkMkQIhANGo"
          + "UuD+JvjFTRdESfKO5428e1HAQL412Sa5e16D4E3M\","
          + "\"sha256_root_hash\":\"jdH9k+\\/lb9abMz3N8r7v55+nSAXej3hqPg=\","
          + "\"tree_size\":4301837}";

  public static final String JSON_RESPONSE = ""
      + "{\"sct_version\":0,\"id\":\"pLkJkLQYWBSHuxOizGdwCjw1mAT5G9+443fNDsgN3BA=\","
      + "\"timestamp\":1373015623951,\n"
      + "\"extensions\":\"\",\n"
      + "\"signature\":\"BAMARjBEAiAggPtKUMFZ4zmNnPhc7As7VR1Dedsdggs9a8pSEHoyGAIgKGsvIPDZg"
      + "DnxTjGY8fSBwkl15dA0TUqW5ex2HCU7yE8=\"}";

  public static final String LOG_ENTRY = "{ \"entries\": [ { \"leaf_input\": \"AAAAAAFHz32CRgAAAALO"
      + "MIICyjCCAjOgAwIBAgIBBjANBgkqhkiG9w0BAQUFADBVMQswCQYDVQQGEwJHQjEkMCIGA1UEChMbQ2VydGlmaWNhdG"
      + "UgVHJhbnNwYXJlbmN5IENBMQ4wDAYDVQQIEwVXYWxlczEQMA4GA1UEBxMHRXJ3IFdlbjAeFw0xMjA2MDEwMDAwMDBa"
      + "Fw0yMjA2MDEwMDAwMDBaMFIxCzAJBgNVBAYTAkdCMSEwHwYDVQQKExhDZXJ0aWZpY2F0ZSBUcmFuc3BhcmVuY3kxDj"
      + "AMBgNVBAgTBVdhbGVzMRAwDgYDVQQHEwdFcncgV2VuMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCx+jeTYRH4"
      + "eS2iCBw\\/5BklAIUx3H8sZXvZ4d5HBBYLTJ8Z1UraRHBATBxRNBuPH3U43d0o2aykg2n8VkbdzHYX+BaKrltB1DM"
      + "x\\/KLa38gE1XIIlJBh+e75AspHzojGROAA8G7uzKvcndL2iiLMsJ3Hbg28c1J3ZbGjeoxnYlPcwQIDAQABo4GsMIG"
      + "pMB0GA1UdDgQWBBRqDZgqO2LES20u9Om7egGqnLeY4jB9BgNVHSMEdjB0gBRfnYgNyHPmVNT4DdjmsMEktEfDVaFZp"
      + "FcwVTELMAkGA1UEBhMCR0IxJDAiBgNVBAoTG0NlcnRpZmljYXRlIFRyYW5zcGFyZW5jeSBDQTEOMAwGA1UECBMFV2F"
      + "sZXMxEDAOBgNVBAcTB0VydyBXZW6CAQAwCQYDVR0TBAIwADANBgkqhkiG9w0BAQUFAAOBgQAXHNhKrEFKmgMPIqrI9"
      + "oiwgbJwm4SLTlURQGzXB\\/7QKFl6n678Lu4peNYzqqwU7TI1GX2ofg9xuIdfGsnniygXSd3t0Afj7PUGRfjL9mclb"
      + "NahZHteEyA7uFgt59Zpb2VtHGC5X0Vrf88zhXGQjxxpcn0kxPzNJJKVeVgU0drA5gAA\", "
      + "\"extra_data\": \"AALXAALUMIIC0DCCAjmgAwIBAgIBADANBgkqhkiG9w0BAQUFADBVMQswCQYDVQQGEwJHQjEk"
      + "MCIGA1UEChMbQ2VydGlmaWNhdGUgVHJhbnNwYXJlbmN5IENBMQ4wDAYDVQQIEwVXYWxlczEQMA4GA1UEBxMHRXJ3IF"
      + "dlbjAeFw0xMjA2MDEwMDAwMDBaFw0yMjA2MDEwMDAwMDBaMFUxCzAJBgNVBAYTAkdCMSQwIgYDVQQKExtDZXJ0aWZp"
      + "Y2F0ZSBUcmFuc3BhcmVuY3kgQ0ExDjAMBgNVBAgTBVdhbGVzMRAwDgYDVQQHEwdFcncgV2VuMIGfMA0GCSqGSIb3DQ"
      + "EBAQUAA4GNADCBiQKBgQDVimhTYhCicRmTbneDIRgcKkATxtB7jHbrkVfT0PtLO1FuzsvRyY2RxS90P6tjXVUJnNE"
      + "6uvMa5UFEJFGnTHgW8iQ8+EjPKDHM5nugSlojgZ88ujfmJNnDvbKZuDnd\\/iYx0ss6hPx7srXFL8\\/BT\\/9Ab1z"
      + "URmnLsvfP34b7arnRsQIDAQABo4GvMIGsMB0GA1UdDgQWBBRfnYgNyHPmVNT4DdjmsMEktEfDVTB9BgNVHSMEdjB0g"
      + "BRfnYgNyHPmVNT4DdjmsMEktEfDVaFZpFcwVTELMAkGA1UEBhMCR0IxJDAiBgNVBAoTG0NlcnRpZmljYXRlIFRyYW5"
      + "zcGFyZW5jeSBDQTEOMAwGA1UECBMFV2FsZXMxEDAOBgNVBAcTB0VydyBXZW6CAQAwDAYDVR0TBAUwAwEB\\/zANBgk"
      + "qhkiG9w0BAQUFAAOBgQAGCMxKbWTyIF4UbASydvkrDvqUpdryOvw4BmBtOZDQoeojPUApV2lGOwRmYef6HReZFSCa6"
      + "i4Kd1F2QRIn18ADB8dHDmFYT9czQiRyf1HWkLxHqd81TbD26yWVXeGJPE3VICskovPkQNJ0tU4b03YmnKliibduyqQ"
      + "QkOFPOwqULg==\" } ] }";
  
  public static final String LOG_ENTRY_AND_PROOF = "{\"leaf_input\": \"AAAAAAFHz32"
      + "CRgAAAALO"
      + "MIICyjCCAjOgAwIBAgIBBjANBgkqhkiG9w0BAQUFADBVMQswCQYDVQQGEwJHQjEkMCIGA1UEChMbQ2VydGlmaWNhdG"
      + "UgVHJhbnNwYXJlbmN5IENBMQ4wDAYDVQQIEwVXYWxlczEQMA4GA1UEBxMHRXJ3IFdlbjAeFw0xMjA2MDEwMDAwMDBa"
      + "Fw0yMjA2MDEwMDAwMDBaMFIxCzAJBgNVBAYTAkdCMSEwHwYDVQQKExhDZXJ0aWZpY2F0ZSBUcmFuc3BhcmVuY3kxDj"
      + "AMBgNVBAgTBVdhbGVzMRAwDgYDVQQHEwdFcncgV2VuMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCx+jeTYRH4"
      + "eS2iCBw\\/5BklAIUx3H8sZXvZ4d5HBBYLTJ8Z1UraRHBATBxRNBuPH3U43d0o2aykg2n8VkbdzHYX+BaKrltB1DM"
      + "x\\/KLa38gE1XIIlJBh+e75AspHzojGROAA8G7uzKvcndL2iiLMsJ3Hbg28c1J3ZbGjeoxnYlPcwQIDAQABo4GsMIG"
      + "pMB0GA1UdDgQWBBRqDZgqO2LES20u9Om7egGqnLeY4jB9BgNVHSMEdjB0gBRfnYgNyHPmVNT4DdjmsMEktEfDVaFZp"
      + "FcwVTELMAkGA1UEBhMCR0IxJDAiBgNVBAoTG0NlcnRpZmljYXRlIFRyYW5zcGFyZW5jeSBDQTEOMAwGA1UECBMFV2F"
      + "sZXMxEDAOBgNVBAcTB0VydyBXZW6CAQAwCQYDVR0TBAIwADANBgkqhkiG9w0BAQUFAAOBgQAXHNhKrEFKmgMPIqrI9"
      + "oiwgbJwm4SLTlURQGzXB\\/7QKFl6n678Lu4peNYzqqwU7TI1GX2ofg9xuIdfGsnniygXSd3t0Afj7PUGRfjL9mclb"
      + "NahZHteEyA7uFgt59Zpb2VtHGC5X0Vrf88zhXGQjxxpcn0kxPzNJJKVeVgU0drA5gAA\", "
      + "\"extra_data\": \"AALXAALUMIIC0DCCAjmgAwIBAgIBADANBgkqhkiG9w0BAQUFADBVMQswCQYDVQQGEwJHQjEk"
      + "MCIGA1UEChMbQ2VydGlmaWNhdGUgVHJhbnNwYXJlbmN5IENBMQ4wDAYDVQQIEwVXYWxlczEQMA4GA1UEBxMHRXJ3IF"
      + "dlbjAeFw0xMjA2MDEwMDAwMDBaFw0yMjA2MDEwMDAwMDBaMFUxCzAJBgNVBAYTAkdCMSQwIgYDVQQKExtDZXJ0aWZp"
      + "Y2F0ZSBUcmFuc3BhcmVuY3kgQ0ExDjAMBgNVBAgTBVdhbGVzMRAwDgYDVQQHEwdFcncgV2VuMIGfMA0GCSqGSIb3DQ"
      + "EBAQUAA4GNADCBiQKBgQDVimhTYhCicRmTbneDIRgcKkATxtB7jHbrkVfT0PtLO1FuzsvRyY2RxS90P6tjXVUJnNE"
      + "6uvMa5UFEJFGnTHgW8iQ8+EjPKDHM5nugSlojgZ88ujfmJNnDvbKZuDnd\\/iYx0ss6hPx7srXFL8\\/BT\\/9Ab1z"
      + "URmnLsvfP34b7arnRsQIDAQABo4GvMIGsMB0GA1UdDgQWBBRfnYgNyHPmVNT4DdjmsMEktEfDVTB9BgNVHSMEdjB0g"
      + "BRfnYgNyHPmVNT4DdjmsMEktEfDVaFZpFcwVTELMAkGA1UEBhMCR0IxJDAiBgNVBAoTG0NlcnRpZmljYXRlIFRyYW5"
      + "zcGFyZW5jeSBDQTEOMAwGA1UECBMFV2FsZXMxEDAOBgNVBAcTB0VydyBXZW6CAQAwDAYDVR0TBAUwAwEB\\/zANBgk"
      + "qhkiG9w0BAQUFAAOBgQAGCMxKbWTyIF4UbASydvkrDvqUpdryOvw4BmBtOZDQoeojPUApV2lGOwRmYef6HReZFSCa6"
      + "i4Kd1F2QRIn18ADB8dHDmFYT9czQiRyf1HWkLxHqd81TbD26yWVXeGJPE3VICskovPkQNJ0tU4b03YmnKliibduyqQ"
      + "QkOFPOwqULg==\", "
      + "\"audit_path\":[\"h6Wo6zvO+d293qbd/5bfwMae9eh4jAZULr6i2fLAop4=\","
      + "\"6eIbVFV8aYnfVF4/S3JN+DMPqjzBHyEMooN3rIkGbC4=\"] }";
  
  
  public static final String LOG_ENTRY_CORRUPTED_ENTRY =
      "{ \"entries\": [ { \"leaf_input\": \"AAAAAAFHz32CRgAAAALO"
      + "MIICyjCCAjOgAwIBAgIBBjANBgkqhkiG9w0BAQUFADBVMQswCQYDVQQGEwJHQjEkMCIGA1UEChMbQ2VydGlmaWNhdG"
      + "UgVHJhbnNwYXJlbmN5IENBMQ4wDAYDVQQIEwVXYWxlczEQMA4GA1UEBxMHRXJ3IFdlbjAeFw0xMjA2MDEwMDAwMDBa"
      + "Fw0yMjA2MDEwMDAwMDBaMFIxCzAJBgNVBAYTAkdCMSEwHwYDVQQKExhDZXJ0aWZpY2F0ZSBUcmFuc3BhcmVuY3kxDj"
      + "AMBgNVBAgTBVdhbGVzMRAwDgYDVQQHEwdFcncgV2VuMIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCx+jeTYRH4"
      + "eS2iCBw\\/5BklAIUx3H8sZXvZ4d5HBBYLTJ8Z1UraRHBATBxRNBuPH3U43d0o2aykg2n8VkbdzHYX+BaKrltB1DM"
      + "x\\/KLa38gE1XIIlJBh+e75AspHzojGROAA8G7uzKvcndL2iiLMsJ3Hbg28c1J3ZbGjeoxnYlPcwQIDAQABo4GsMIG"
      + "FcwVTELMAkGA1UEBhMCR0IxJDAiBgNVBAoTG0NlcnRpZmljYXRlIFRyYW5zcGFyZW5jeSBDQTEOMAwGA1UECBMFV2F"
      + "sZXMxEDAOBgNVBAcTB0VydyBXZW6CAQAwCQYDVR0TBAIwADANBgkqhkiG9w0BAQUFAAOBgQAXHNhKrEFKmgMPIqrI9"
      + "oiwgbJwm4SLTlURQGzXB\\/7QKFl6n678Lu4peNYzqqwU7TI1GX2ofg9xuIdfGsnniygXSd3t0Afj7PUGRfjL9mclb"
      + "NahZHteEyA7uFgt59Zpb2VtHGC5X0Vrf88zhXGQjxxpcn0kxPzNJJKVeVgU0drA5gAA\", "
      + "\"extra_data\": \"AALXAALUMIIC0DCCAjmgAwIBAgIBADANBgkqhkiG9w0BAQUFADBVMQswCQYDVQQGEwJHQjEk"
      + "MCIGA1UEChMbQ2VydGlmaWNhdGUgVHJhbnNwYXJlbmN5IENBMQ4wDAYDVQQIEwVXYWxlczEQMA4GA1UEBxMHRXJ3IF"
      + "dlbjAeFw0xMjA2MDEwMDAwMDBaFw0yMjA2MDEwMDAwMDBaMFUxCzAJBgNVBAYTAkdCMSQwIgYDVQQKExtDZXJ0aWZp"
      + "Y2F0ZSBUcmFuc3BhcmVuY3kgQ0ExDjAMBgNVBAgTBVdhbGVzMRAwDgYDVQQHEwdFcncgV2VuMIGfMA0GCSqGSIb3DQ"
      + "EBAQUAA4GNADCBiQKBgQDVimhTYhCicRmTbneDIRgcKkATxtB7jHbrkVfT0PtLO1FuzsvRyY2RxS90P6tjXVUJnNE"
      + "6uvMa5UFEJFGnTHgW8iQ8+EjPKDHM5nugSlojgZ88ujfmJNnDvbKZuDnd\\/iYx0ss6hPx7srXFL8\\/BT\\/9Ab1z"
      + "URmnLsvfP34b7arnRsQIDAQABo4GvMIGsMB0GA1UdDgQWBBRfnYgNyHPmVNT4DdjmsMEktEfDVTB9BgNVHSMEdjB0g"
      + "BRfnYgNyHPmVNT4DdjmsMEktEfDVaFZpFcwVTELMAkGA1UEBhMCR0IxJDAiBgNVBAoTG0NlcnRpZmljYXRlIFRyYW5"
      + "zcGFyZW5jeSBDQTEOMAwGA1UECBMFV2FsZXMxEDAOBgNVBAcTB0VydyBXZW6CAQAwDAYDVR0TBAUwAwEB\\/zANBgk"
      + "qhkiG9w0BAQUFAAOBgQAGCMxKbWTyIF4UbASydvkrDvqUpdryOvw4BmBtOZDQoeojPUApV2lGOwRmYef6HReZFSCa6"
      + "i4Kd1F2QRIn18ADB8dHDmFYT9czQiRyf1HWkLxHqd81TbD26yWVXeGJPE3VICskovPkQNJ0tU4b03YmnKliibduyqQ"
      + "QkOFPOwqULg==\" } ] }";
  
  public static final String LOG_ENTRY_EMPTY = "{ \"entries\": []}";
  
  public static final String CONSISTENCY_PROOF = "{\"consistency\" :[\"wDblrkBlhZ7UqimOaRS18MjqvNyt"
      + "/Fc2tcy6nWONY84=\",\"/NeD2RVJUnnzreBeKM4fCCWk+KZzG2ctHdm9LLngwJY=\"]}";
  
  public static final String CONSISTENCY_PROOF_EMPTY = "{ \"consistency\": []}";

  public static final byte[] LOG_ID =
      Base64.decodeBase64("pLkJkLQYWBSHuxOizGdwCjw1mAT5G9+443fNDsgN3BA=");

  @Test
  public void certificatesAreEncoded() throws CertificateException, IOException {
    List<Certificate> inputCerts = CryptoDataLoader.certificatesFromFile(new File(TEST_DATA_PATH));
    HttpLogClient client = new HttpLogClient("");

    JSONObject encoded = client.encodeCertificates(inputCerts);
    Assert.assertTrue(encoded.containsKey("chain"));
    JSONArray chain = (JSONArray) encoded.get("chain");
    assertEquals("Expected to have two certificates in the chain", 2, chain.size());
    // Make sure the order is reversed.
    for (int i = 0; i < inputCerts.size(); i++) {
      assertEquals(
          Base64.encodeBase64String(inputCerts.get(i).getEncoded()),
          chain.get(i));
    }
  }

  public void verifySCTContents(Ct.SignedCertificateTimestamp sct) {
    assertEquals(Ct.Version.V1, sct.getVersion());
    assertArrayEquals(LOG_ID, sct.getId().getKeyId().toByteArray());
    assertEquals(1373015623951L, sct.getTimestamp());
    assertEquals(Ct.DigitallySigned.HashAlgorithm.SHA256, sct.getSignature().getHashAlgorithm());
    assertEquals(Ct.DigitallySigned.SignatureAlgorithm.ECDSA, sct.getSignature().getSigAlgorithm());
  }
  @Test
  public void serverResponseParsed() throws IOException {
    Ct.SignedCertificateTimestamp sct = HttpLogClient.parseServerResponse(JSON_RESPONSE);
    verifySCTContents(sct);
  }

  @Test
  public void certificateSentToServer() throws IOException, CertificateException {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    when(mockInvoker.makePostRequest(eq("http://ctlog/add-chain"), Matchers.anyString()))
      .thenReturn(JSON_RESPONSE);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    List<Certificate> certs = CryptoDataLoader.certificatesFromFile(new File(TEST_DATA_PATH));
    Ct.SignedCertificateTimestamp res = client.addCertificate(certs);
    assertNotNull("Should have a meaningful SCT", res);

    verifySCTContents(res);
  }
  
  @Test
  public void getLogSTH() throws IllegalAccessException, IllegalArgumentException,
    InvocationTargetException, NoSuchMethodException, SecurityException {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-sth"))).thenReturn(STH_RESPONSE);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    Ct.SignedTreeHead sth = client.getLogSTH();

    Assert.assertNotNull(sth);
    Assert.assertEquals(1402415255382L, sth.getTimestamp());
    Assert.assertEquals(4301837, sth.getTreeSize());
    String rootHash = Base64.encodeBase64String(sth.getSha256RootHash().toByteArray());
    Assert.assertTrue("jdH9k+/lb9abMz3N8rVmwrw8MWU7v55+nSAXej3hqPg=".equals(rootHash));
  }
  
  @Test
  public void getLogSTHBadResponseTimestamp() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-sth"))).thenReturn(
        BAD_STH_RESPONSE_INVALID_TIMESTAMP);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    try {
      client.getLogSTH();
      Assert.fail();
    } catch (CertificateTransparencyException e) {
    }
  }
  
  @Test
  public void getLogSTHBadResponseRootHash() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-sth"))).thenReturn(
        BAD_STH_RESPONSE_INVALID_ROOT_HASH);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    try {
      client.getLogSTH();
      Assert.fail();
    } catch (CertificateTransparencyException e) {
    }
  }
  
  @Test
  public void getRootCerts() throws IOException, ParseException {
    JSONParser parser = new JSONParser();
    Object obj = parser.parse(new FileReader(TestData.TEST_ROOT_CERTS));
    JSONObject response = (JSONObject) obj;
    
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-roots"))).thenReturn(
        response.toJSONString());

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    List<Certificate> rootCerts = client.getLogRoots();

    Assert.assertNotNull(rootCerts);
    Assert.assertEquals(2, rootCerts.size());
  }

  @Test
  public void getLogEntries() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    List<NameValuePair> params = new ArrayList<NameValuePair>();
    params.add(new BasicNameValuePair("start", Long.toString(0)));
    params.add(new BasicNameValuePair("end", Long.toString(0)));

    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-entries"), eq(params)))
      .thenReturn(LOG_ENTRY);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    X509Certificate testChainCert= (X509Certificate) CryptoDataLoader.certificatesFromFile(
        new File(TestData.ROOT_CA_CERT)).get(0);
    X509Certificate testCert= (X509Certificate) CryptoDataLoader.certificatesFromFile(
        new File(TestData.TEST_CERT)).get(0);
    List<ParsedLogEntry> entries = client.getLogEntries(0, 0);

    X509Certificate  chainCert = null ;
    X509Certificate leafCert = null ;
    try {

      byte[] leafCertBytes = entries.get(0).getLogEntry().getX509Entry().getLeafCertificate()
        .toByteArray();
      leafCert = (X509Certificate) CertificateFactory.getInstance("X509").generateCertificate(
        new ByteArrayInputStream(leafCertBytes));

      byte[] chainCertBytes = entries.get(0).getLogEntry().getX509Entry().getCertificateChain(0)
        .toByteArray();
      chainCert = (X509Certificate) CertificateFactory.getInstance("X509").generateCertificate(
        new ByteArrayInputStream(chainCertBytes));
  
    } catch (CertificateException  e) {
      Assert.fail();
    }
    Assert.assertTrue(testCert.equals(leafCert));;
    Assert.assertTrue(testChainCert.equals(chainCert));;
  }

  @Test
  public void getLogEntriesCorruptedEntry() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    List<NameValuePair> params = new ArrayList<NameValuePair>();
    params.add(new BasicNameValuePair("start", Long.toString(0)));
    params.add(new BasicNameValuePair("end", Long.toString(0)));

    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-entries"), eq(params)))
      .thenReturn(LOG_ENTRY_CORRUPTED_ENTRY);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    try {
      // Must get an actual entry as the list of entries is lazily transformed.
      client.getLogEntries(0, 0).get(0);
      Assert.fail();
    } catch (CertificateTransparencyException expected) {
    }
  }

  @Test
  public void getLogEntriesEmptyEntry() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    List<NameValuePair> params = new ArrayList<NameValuePair>();
    params.add(new BasicNameValuePair("start", Long.toString(0)));
    params.add(new BasicNameValuePair("end", Long.toString(0)));

    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-entries"), eq(params)))
      .thenReturn(LOG_ENTRY_EMPTY);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    Assert.assertTrue(client.getLogEntries(0, 0).isEmpty());
  }

  @Test
  public void getSTHConsistency() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    List<NameValuePair> params = new ArrayList<NameValuePair>();
    params.add(new BasicNameValuePair("first", Long.toString(1)));
    params.add(new BasicNameValuePair("second", Long.toString(3)));

    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-sth-consistency"), eq(params)))
      .thenReturn(CONSISTENCY_PROOF);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    List<ByteString> proof = client.getSTHConsistency(1, 3);
    Assert.assertNotNull(proof);
    Assert.assertEquals(2, proof.size());
  }

  @Test
  public void getSTHConsistencyEmpty() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    List<NameValuePair> params = new ArrayList<NameValuePair>();
    params.add(new BasicNameValuePair("first", Long.toString(1)));
    params.add(new BasicNameValuePair("second", Long.toString(3)));

    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-sth-consistency"), eq(params)))
      .thenReturn(CONSISTENCY_PROOF_EMPTY);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    List<ByteString> proof = client.getSTHConsistency(1, 3);
    Assert.assertNotNull(proof);
    Assert.assertTrue(proof.isEmpty());
  }

  @Test
  public void getLogEntrieAndProof() {
    HttpInvoker mockInvoker = mock(HttpInvoker.class);
    List<NameValuePair> params = new ArrayList<NameValuePair>();
    params.add(new BasicNameValuePair("leaf_index", Long.toString(1)));
    params.add(new BasicNameValuePair("tree_size", Long.toString(2)));

    when(mockInvoker.makeGetRequest(eq("http://ctlog/get-entry-and-proof"), eq(params)))
      .thenReturn(LOG_ENTRY_AND_PROOF);

    HttpLogClient client = new HttpLogClient("http://ctlog/", mockInvoker);
    X509Certificate testChainCert= (X509Certificate) CryptoDataLoader.certificatesFromFile(
        new File(TestData.ROOT_CA_CERT)).get(0);
    X509Certificate testCert= (X509Certificate) CryptoDataLoader.certificatesFromFile(
        new File(TestData.TEST_CERT)).get(0);
    ParsedLogEntryWithProof entry = client.getLogEntryAndProof(1, 2);

    X509Certificate  chainCert = null;
    X509Certificate leafCert = null;
    try {

      byte[] leafCertBytes = entry.getParsedLogEntry().getLogEntry().getX509Entry()
        .getLeafCertificate().toByteArray();
      leafCert = (X509Certificate) CertificateFactory.getInstance("X509").generateCertificate(
        new ByteArrayInputStream(leafCertBytes));

      byte[] chainCertBytes = entry.getParsedLogEntry().getLogEntry().getX509Entry()
        .getCertificateChain(0).toByteArray();
      chainCert = (X509Certificate) CertificateFactory.getInstance("X509").generateCertificate(
        new ByteArrayInputStream(chainCertBytes));
  
    } catch (CertificateException  e) {
      Assert.fail();
    }

    Assert.assertTrue(testCert.equals(leafCert));
    Assert.assertTrue(testChainCert.equals(chainCert));
    Assert.assertEquals(2, entry.getAuditProof().getPathNodeList().size());
    Assert.assertEquals(1, entry.getAuditProof().getLeafIndex());
    Assert.assertEquals(2, entry.getAuditProof().getTreeSize());
  }
}
