package org.certificatetransparency.ctlog.utils;

import org.certificatetransparency.ctlog.serialization.CryptoDataLoader;

import java.io.File;
import java.io.IOException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.util.List;

/**
 * Utility class for printing certificate chains. Openssl is probably better for this.
 */
public class PrintCertificates {
  public static void main(String[] args) throws IOException, CertificateException {
    if (args.length < 1) {
      System.out.println(String.format("Usage: %s <certificate chain>",
          PrintCertificates.class.getSimpleName()));
      return;
    }

    String pemFile = args[0];

    List<Certificate> certs = CryptoDataLoader.certificatesFromFile(new File(pemFile));

    System.out.println(String.format("Total number of certificates in chain: %d", certs.size()));
    for (Certificate cert : certs) {
      System.out.println("------------------------------------------");
      System.out.println(cert);
    }
  }
}
