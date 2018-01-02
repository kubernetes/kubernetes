package org.certificatetransparency.ctlog;

import com.google.common.io.Files;
import org.certificatetransparency.ctlog.comm.HttpLogClient;
import org.certificatetransparency.ctlog.proto.Ct;
import org.certificatetransparency.ctlog.serialization.CryptoDataLoader;
import org.certificatetransparency.ctlog.serialization.Serializer;

import java.io.File;
import java.io.IOException;
import java.security.cert.Certificate;
import java.util.List;

/**
 * The main CT log client. Currently only knows how to upload certificate chains
 * to the ctlog.
 */
public class CTLogClient {
  private final HttpLogClient httpClient;
  private final LogSignatureVerifier signatureVerifier;

  /**
   * Result of the certificate upload. Contains the SCT and verification result.
   */
  public static class UploadResult {
    private final Ct.SignedCertificateTimestamp sct;
    private final boolean verified;

    public UploadResult(Ct.SignedCertificateTimestamp sct, boolean verified) {
      this.sct = sct;
      this.verified = verified;
    }

    public boolean isVerified() {
      return verified;
    }

    public final Ct.SignedCertificateTimestamp getSct() {
      return sct;
    }
  }

  public CTLogClient(String baseLogUrl, LogInfo logInfo) {
    this.httpClient = new HttpLogClient(baseLogUrl);
    this.signatureVerifier = new LogSignatureVerifier(logInfo);
  }

  public UploadResult uploadCertificatesChain(List<Certificate> chain) {
    Ct.SignedCertificateTimestamp sct = httpClient.addCertificate(chain);
    return new UploadResult(sct, signatureVerifier.verifySignature(sct, chain.get(0)));
  }

  public static void main(String[] args) throws IOException {
    if (args.length < 3) {
      System.out.println(
          String.format("Usage: %s <Certificate chain> <Log URL> <Log public key> [output file]",
          CTLogClient.class.getSimpleName()));
      return;
    }

    String pemFile = args[0];
    String logUrl = getBaseUrl(args[1]);
    String logPublicKeyFile = args[2];
    String outputSctFile = null;
    if (args.length >= 4) {
      outputSctFile = args[3];
    }

    CTLogClient client = new CTLogClient(logUrl, LogInfo.fromKeyFile(logPublicKeyFile));
    List<Certificate> certs = CryptoDataLoader.certificatesFromFile(new File(pemFile));
    System.out.println(String.format("Total number of certificates: %d", certs.size()));

    UploadResult result = client.uploadCertificatesChain(certs);
    if (result.isVerified()) {
      System.out.println("Upload successful ");
      if (outputSctFile != null) {
        byte[] serialized = Serializer.serializeSctToBinary(result.getSct());
        Files.write(serialized, new File(outputSctFile));
      }
    } else {
      System.out.println("Log signature verification FAILED.");
    }
  }

  private static String getBaseUrl(String url) {
    return String.format("http://%s/ct/v1/", url);
  }
}
