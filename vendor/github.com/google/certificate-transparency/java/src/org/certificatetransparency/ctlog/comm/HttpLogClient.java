package org.certificatetransparency.ctlog.comm;

import java.io.ByteArrayInputStream;
import java.security.cert.Certificate;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.codec.binary.Base64;
import org.apache.http.NameValuePair;
import org.apache.http.message.BasicNameValuePair;
import org.certificatetransparency.ctlog.CertificateInfo;
import org.certificatetransparency.ctlog.CertificateTransparencyException;
import org.certificatetransparency.ctlog.ParsedLogEntry;
import org.certificatetransparency.ctlog.ParsedLogEntryWithProof;
import org.certificatetransparency.ctlog.proto.Ct;
import org.certificatetransparency.ctlog.serialization.Deserializer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.protobuf.ByteString;

/**
 * A CT HTTP client. Abstracts away the json encoding necessary for the server.
 */
public class HttpLogClient {
  private static final String ADD_PRE_CHAIN_PATH = "add-pre-chain";
  private static final String ADD_CHAIN_PATH = "add-chain";
  private static final String GET_STH_PATH = "get-sth";
  private static final String GET_ROOTS_PATH = "get-roots";
  private static final String GET_ENTRIES = "get-entries";
  private static final String GET_STH_CONSISTENCY = "get-sth-consistency";
  private static final String GET_ENTRY_AND_PROOF = "get-entry-and-proof";

  private final String logUrl;
  private final HttpInvoker postInvoker;

  /**
   * New HttpLogClient.
   * @param logUrl CT Log's full URL, e.g. "http://ct.googleapis.com/pilot/ct/v1/"
   */
  public HttpLogClient(String logUrl) {
    this(logUrl, new HttpInvoker());
  }

  /**
   * For testing specify an HttpInvoker
   * @param logUrl URL of the log.
   * @param postInvoker HttpInvoker instance to use.
   */
  public HttpLogClient(String logUrl, HttpInvoker postInvoker) {
    this.logUrl = logUrl;
    this.postInvoker = postInvoker;
  }

  /**
   * JSON-encodes the list of certificates into a JSON object.
   * @param certs Certificates to encode.
   * @return A JSON object with one field, "chain", holding a JSON array of base64-encoded certs.
   */
  @SuppressWarnings("unchecked") // Because JSONObject, JSONArray extend raw types.
  JSONObject encodeCertificates(List<Certificate> certs) {
    JSONArray retObject = new JSONArray();

    try {
      for (Certificate cert : certs) {
        retObject.add(Base64.encodeBase64String(cert.getEncoded()));
      }
    } catch (CertificateEncodingException e) {
      throw new CertificateTransparencyException("Error encoding certificate", e);
    }

    JSONObject jsonObject = new JSONObject();
    jsonObject.put("chain", retObject);
    return jsonObject;
  }

  /**
   * Parses the CT Log's json response into a proper proto.
   *
   * @param responseBody Response string to parse.
   * @return SCT filled from the JSON input.
   */
  static Ct.SignedCertificateTimestamp parseServerResponse(String responseBody) {
    if (responseBody == null) {
      return null;
    }

    JSONObject parsedResponse = (JSONObject) JSONValue.parse(responseBody);
    Ct.SignedCertificateTimestamp.Builder builder =
        Ct.SignedCertificateTimestamp.newBuilder();

    int numericVersion = ((Number) parsedResponse.get("sct_version")).intValue();
    Ct.Version version = Ct.Version.valueOf(numericVersion);
    if (version == null) {
      throw new IllegalArgumentException(String.format("Input JSON has an invalid version: %d",
          numericVersion));
    }
    builder.setVersion(version);
    Ct.LogID.Builder logIdBuilder = Ct.LogID.newBuilder();
    logIdBuilder.setKeyId(
        ByteString.copyFrom(Base64.decodeBase64((String) parsedResponse.get("id"))));
    builder.setId(logIdBuilder.build());
    builder.setTimestamp(((Number) parsedResponse.get("timestamp")).longValue());
    String extensions = (String) parsedResponse.get("extensions");
    if (!extensions.isEmpty()) {
      builder.setExtensions(ByteString.copyFrom(Base64.decodeBase64(extensions)));
    }

    String base64Signature = (String) parsedResponse.get("signature");
    builder.setSignature(
        Deserializer.parseDigitallySignedFromBinary(
            new ByteArrayInputStream(Base64.decodeBase64(base64Signature))));
    return builder.build();
  }

  /**
   * Adds a certificate to the log.
   * @param certificatesChain The certificate chain to add.
   * @return SignedCertificateTimestamp if the log added the chain successfully.
   */
  public Ct.SignedCertificateTimestamp addCertificate(List<Certificate> certificatesChain) {
    Preconditions.checkArgument(!certificatesChain.isEmpty(),
        "Must have at least one certificate to submit.");

    boolean isPreCertificate = CertificateInfo.isPreCertificate(certificatesChain.get(0));
    if (isPreCertificate &&
        CertificateInfo.isPreCertificateSigningCert(certificatesChain.get(1))) {
      Preconditions.checkArgument(
          certificatesChain.size() >= 3,
          "When signing a PreCertificate with a PreCertificate Signing Cert," +
              " the issuer certificate must follow.");
    }

    return addCertificate(certificatesChain, isPreCertificate);
  }

  private Ct.SignedCertificateTimestamp addCertificate(
      List<Certificate> certificatesChain, boolean isPreCertificate) {
    String jsonPayload = encodeCertificates(certificatesChain).toJSONString();
    String methodPath;
    if (isPreCertificate) {
      methodPath = ADD_PRE_CHAIN_PATH;
    } else {
      methodPath = ADD_CHAIN_PATH;
    }

    String response = postInvoker.makePostRequest(logUrl + methodPath, jsonPayload);
    return parseServerResponse(response);
  }
  
  /**
   * Retrieves Latest Signed Tree Head from the log.
   * The signature of the Signed Tree Head component is not verified.
   * @return latest STH
   */
  public Ct.SignedTreeHead getLogSTH() {
    String response = postInvoker.makeGetRequest(logUrl + GET_STH_PATH);
    return parseSTHResponse(response);
  }

  /**
   * Retrieves accepted Root Certificates.
   * @return a list of root certificates.
   */
  public List<Certificate> getLogRoots() {
    String response = postInvoker.makeGetRequest(logUrl + GET_ROOTS_PATH);

    return parseRootCertsResponse(response);
  }

  /**
   * Retrieve Entries from Log.
   * @param start 0-based index of first entry to retrieve, in decimal.
   * @param end 0-based index of last entry to retrieve, in decimal.
   * @return list of Log's entries.
   */
  public List<ParsedLogEntry> getLogEntries(long start, long end) {
    Preconditions.checkArgument(0 <= start && end >= start);

    List<NameValuePair> params = createParamsList("start", "end", Long.toString(start),
      Long.toString(end));

    String response = postInvoker.makeGetRequest(logUrl + GET_ENTRIES, params);
    return parseLogEntries(response);
  }

  /**
   * Retrieve Merkle Consistency Proof between Two Signed Tree Heads.
   * @param first The tree_size of the first tree, in decimal.
   * @param second The tree_size of the second tree, in decimal.
   * @return A list of base64 decoded Merkle Tree nodes serialized to ByteString objects.
   */
  public List<ByteString> getSTHConsistency(long first, long second) {
    Preconditions.checkArgument(0 <= first && second >= first);

    List<NameValuePair> params = createParamsList("first", "second", Long.toString(first),
      Long.toString(second));

    String response = postInvoker.makeGetRequest(logUrl + GET_STH_CONSISTENCY, params);
    return parseConsistencyProof(response);
  }

  /**
   * Retrieve Entry+Merkle Audit Proof from Log.
   * @param leaf_index The index of the desired entry.
   * @param tree_size The tree_size of the tree for which the proof is desired.
   * @return ParsedLog entry object with proof.
   */
  public ParsedLogEntryWithProof getLogEntryAndProof(long leafindex, long treeSize) {
    Preconditions.checkArgument(0 <= leafindex && treeSize >= leafindex);

    List<NameValuePair> params = createParamsList("leaf_index", "tree_size",
      Long.toString(leafindex), Long.toString(treeSize));

    String response = postInvoker.makeGetRequest(logUrl + GET_ENTRY_AND_PROOF, params);
    JSONObject entry = (JSONObject) JSONValue.parse(response);
    JSONArray auditPath = (JSONArray) entry.get("audit_path");

    return Deserializer.parseLogEntryWithProof(jsonToLogEntry.apply(entry), auditPath, leafindex,
      treeSize);
  }

  /**
   * Creates a list of NameValuePair objects.
   * @param firstParamName The first parameter name.
   * @param firstParamValue The first parameter value.
   * @param secondParamName The second parameter name.
   * @param secondParamValue The second parameter value.
   * @return A list of NameValuePair objects.
   */
  private List<NameValuePair> createParamsList(String firstParamName, String secondParamName,
    String firstParamValue, String secondParamValue) {
    List<NameValuePair> params = new ArrayList<NameValuePair>();
    params.add(new BasicNameValuePair(firstParamName, firstParamValue));
    params.add(new BasicNameValuePair(secondParamName, secondParamValue));
    return params;
  }

  /**
   * Parses CT log's response for "get-entries" into a list of {@link ParsedLogEntry} objects.
   * @param response Log response to pars.
   * @return list of Log's entries.
   */
  @SuppressWarnings("unchecked")
  private List<ParsedLogEntry> parseLogEntries(String response) {
    Preconditions.checkNotNull(response, "Log entries response from Log should not be null.");

    JSONObject responseJson = (JSONObject) JSONValue.parse(response);
    JSONArray arr = (JSONArray) responseJson.get("entries");
    return Lists.transform(arr, jsonToLogEntry);
  }

  private Function<JSONObject, ParsedLogEntry> jsonToLogEntry =
    new Function<JSONObject, ParsedLogEntry>() {
    @Override public ParsedLogEntry apply(JSONObject entry) {
      String leaf = (String) entry.get("leaf_input");
      String extra = (String) entry.get("extra_data");

      return Deserializer.parseLogEntry(
        new ByteArrayInputStream(Base64.decodeBase64(leaf)),
        new ByteArrayInputStream(Base64.decodeBase64(extra)));
      }
    };

  /**
   * Parses CT log's response for the "get-sth-consistency" request.
   * @param response JsonObject containing an array of Merkle Tree nodes.
   * @return A list of base64 decoded Merkle Tree nodes serialized to ByteString objects.
   */
  private List<ByteString> parseConsistencyProof(String response) {
    Preconditions.checkNotNull(response, "Merkle Consistency response should not be null.");

    JSONObject responseJson = (JSONObject) JSONValue.parse(response);
    JSONArray arr = (JSONArray) responseJson.get("consistency");

    List<ByteString> proof = new ArrayList<ByteString>();
    for(Object node: arr) {
      proof.add(ByteString.copyFrom(Base64.decodeBase64((String) node)));
    }
    return proof;
  }

  /**
   * Parses CT log's response for "get-sth" into a proto object.
   * @param sthResponse Log response to parse
   * @return a proto object of SignedTreeHead type.
   */
  Ct.SignedTreeHead parseSTHResponse(String sthResponse) {
    Preconditions.checkNotNull(
      sthResponse, "Sign Tree Head response from a CT log should not be null");

    JSONObject response = (JSONObject) JSONValue.parse(sthResponse);
    long treeSize = (Long) response.get("tree_size");
    long timeStamp = (Long) response.get("timestamp");
    if (treeSize < 0 || timeStamp < 0) {
      throw new CertificateTransparencyException(
        String.format("Bad response. Size of tree or timestamp cannot be a negative value. "
          + "Log Tree size: %d Timestamp: %d", treeSize, timeStamp));
    }
    String base64Signature = (String) response.get("tree_head_signature");
    String sha256RootHash = (String) response.get("sha256_root_hash");

    Ct.SignedTreeHead.Builder builder =  Ct.SignedTreeHead.newBuilder();
    builder.setVersion(Ct.Version.V1);
    builder.setTreeSize(treeSize);
    builder.setTimestamp(timeStamp);
    builder.setSha256RootHash(ByteString.copyFrom(Base64.decodeBase64(sha256RootHash)));
    builder.setSignature(Deserializer.parseDigitallySignedFromBinary(
      new ByteArrayInputStream(Base64.decodeBase64(base64Signature))));
    if (builder.getSha256RootHash().size() != 32) {
       throw new CertificateTransparencyException(
         String.format("Bad response. The root hash of the Merkle Hash Tree must be 32 bytes. "
           + "The size of the root hash is %d", builder.getSha256RootHash().size()));
      }
    return builder.build();
  }

  /**
   * Parses the response from "get-roots" GET method.
   *
   * @param response JSONObject with certificates to parse.
   * @return a list of root certificates. 
   */
  List<Certificate> parseRootCertsResponse(String response) {
    List<Certificate> certs = new ArrayList<>();

    JSONObject entries = (JSONObject) JSONValue.parse(response);
    JSONArray entriesArray = (JSONArray) entries.get("certificates");

    for(Object i: entriesArray) {
      // We happen to know that JSONArray contains strings.
      byte[] in = Base64.decodeBase64((String) i);
      try {
        certs.add(CertificateFactory.getInstance("X509").generateCertificate(
          new ByteArrayInputStream(in)));
      } catch (CertificateException e) {
        throw new CertificateTransparencyException(
          "Malformed data from a CT log have been received: " + e.getLocalizedMessage(), e);
      }
    }
    return certs;
  }
}
