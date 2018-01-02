package org.certificatetransparency.ctlog;

import static org.certificatetransparency.ctlog.serialization.CTConstants.POISON_EXTENSION_OID;
import static org.certificatetransparency.ctlog.serialization.CTConstants.PRECERTIFICATE_SIGNING_OID;

import java.security.cert.Certificate;
import java.security.cert.CertificateParsingException;
import java.security.cert.X509Certificate;

/**
 * Helper class for finding out all kinds of information about a certificate.
 */
public class CertificateInfo {
  public static boolean isPreCertificateSigningCert(Certificate signerCert) {
    X509Certificate x509SignerCert = (X509Certificate) signerCert;

    try {
      return (x509SignerCert.getExtendedKeyUsage() != null) &&
          (x509SignerCert.getExtendedKeyUsage().contains(PRECERTIFICATE_SIGNING_OID));
    } catch (CertificateParsingException e) {
      throw new CertificateTransparencyException("Error parsing signer cert: " + e.getMessage(),
          e);
    }
  }

  public static boolean isPreCertificate(Certificate certificate) {
    X509Certificate x509PreCertificate = (X509Certificate) certificate;
    return (x509PreCertificate.getCriticalExtensionOIDs() != null) &&
        x509PreCertificate.getCriticalExtensionOIDs().contains(POISON_EXTENSION_OID);
  }
}
