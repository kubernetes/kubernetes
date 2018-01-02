package org.certificatetransparency.ctlog.utils;

import com.google.common.io.Files;
import org.certificatetransparency.ctlog.comm.HttpLogClient;
import org.certificatetransparency.ctlog.proto.Ct;
import org.certificatetransparency.ctlog.serialization.CryptoDataLoader;

import java.io.File;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.spec.InvalidKeySpecException;
import java.util.List;

/**
 * Utility class for uploading a certificate.
 */
public class UploadCertificate {
  public static void main(String[] args)
      throws IOException, CertificateException, InvalidKeySpecException, NoSuchAlgorithmException,
      InvalidKeyException, SignatureException {
    if (args.length < 1) {
      System.out.println(String.format("Usage: %s <certificates chain> [output file]",
          UploadCertificate.class.getSimpleName()));
      return;
    }

    String pemFile = args[0];

    List<Certificate> certs = CryptoDataLoader.certificatesFromFile(new File(pemFile));
    System.out.println(String.format("Total number of certificates in chain: %d", certs.size()));

    HttpLogClient client = new HttpLogClient("http://ct.googleapis.com/pilot/ct/v1/");

    Ct.SignedCertificateTimestamp resp = client.addCertificate(certs);

    System.out.println(resp);
    if (args.length >= 2) {
      String outputFile = args[1];
      //TODO(eranm): Binary encoding compatible with the C++ code.
      Files.write(resp.toByteArray(), new File(outputFile));
    }
  }
}
