package org.certificatetransparency.ctlog;

import static org.certificatetransparency.ctlog.TestData.PRE_CERT_SIGNING_CERT;
import static org.certificatetransparency.ctlog.TestData.ROOT_CA_CERT;
import static org.certificatetransparency.ctlog.TestData.TEST_CERT;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT;
import static org.certificatetransparency.ctlog.TestData.loadCertificates;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.security.cert.Certificate;

/**
 * Make sure the correct info about certificates is provided.
 */
@RunWith(JUnit4.class)
public class CertificateInfoTest {
  @Test
  public void correctlyIdentifiesPreCertificateSigningCert() {
    Certificate preCertificateSigningCert = loadCertificates(PRE_CERT_SIGNING_CERT).get(0);
    Certificate ordinaryCaCert = loadCertificates(ROOT_CA_CERT).get(0);

    assertTrue(CertificateInfo.isPreCertificateSigningCert(preCertificateSigningCert));
    assertFalse(CertificateInfo.isPreCertificateSigningCert(ordinaryCaCert));
  }

  @Test
  public void correctlyIdentifiesPreCertificates() {
    Certificate regularCert = loadCertificates(TEST_CERT).get(0);
    Certificate preCertificate = loadCertificates(TEST_PRE_CERT).get(0);

    assertTrue(CertificateInfo.isPreCertificate(preCertificate));
    assertFalse(CertificateInfo.isPreCertificate(regularCert));
  }

}
