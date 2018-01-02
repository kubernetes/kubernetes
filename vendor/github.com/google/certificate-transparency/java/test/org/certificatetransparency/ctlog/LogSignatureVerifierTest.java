package org.certificatetransparency.ctlog;

import static org.certificatetransparency.ctlog.TestData.INTERMEDIATE_CA_CERT;
import static org.certificatetransparency.ctlog.TestData.PRE_CERT_SIGNING_BY_INTERMEDIATE;
import static org.certificatetransparency.ctlog.TestData.PRE_CERT_SIGNING_CERT;
import static org.certificatetransparency.ctlog.TestData.ROOT_CA_CERT;
import static org.certificatetransparency.ctlog.TestData.TEST_CERT;
import static org.certificatetransparency.ctlog.TestData.TEST_CERT_SCT;
import static org.certificatetransparency.ctlog.TestData.TEST_INTERMEDIATE_CERT;
import static org.certificatetransparency.ctlog.TestData.TEST_INTERMEDIATE_CERT_SCT;
import static org.certificatetransparency.ctlog.TestData.TEST_LOG_KEY;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT_PRECA_SCT;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT_SIGNED_BY_INTERMEDIATE;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT_SIGNED_BY_INTERMEDIATE_SCT;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT_SIGNED_BY_PRECA_CERT;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT_SIGNED_BY_PRECA_INTERMEDIATE;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_CERT_SIGNED_BY_PRECA_INTERMEDIATE_SCT;
import static org.certificatetransparency.ctlog.TestData.TEST_PRE_SCT;
import static org.certificatetransparency.ctlog.TestData.loadCertificates;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.io.Files;

import org.certificatetransparency.ctlog.proto.Ct;
import org.certificatetransparency.ctlog.serialization.Deserializer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SignatureException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.security.spec.InvalidKeySpecException;
import java.util.ArrayList;
import java.util.List;


/**
 * This test verifies that the data is correctly serialized for signature comparison, so
 * signature verification is actually effective.
 */
@RunWith(JUnit4.class)
public class LogSignatureVerifierTest {
  private LogSignatureVerifier getVerifier() {
    LogInfo logInfo = LogInfo.fromKeyFile(TEST_LOG_KEY);
    return new LogSignatureVerifier(logInfo);
  }

  /**
   * Tests for package-visible methods.
   */
  @Test
  public void signatureVerifies() throws IOException, CertificateException,
      InvalidKeySpecException, NoSuchAlgorithmException, SignatureException, InvalidKeyException {
    List<Certificate> certs = loadCertificates(TEST_CERT);
    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(Files.toByteArray(new File(TEST_CERT_SCT))));
    LogSignatureVerifier verifier = getVerifier();
    assertTrue(verifier.verifySignature(sct, certs.get(0)));
  }

  @Test
  public void signatureOnPreCertificateVerifies() throws IOException {
    List<Certificate> preCertificatesList = loadCertificates(TEST_PRE_CERT);
    assertEquals(1, preCertificatesList.size());
    Certificate preCertificate = preCertificatesList.get(0);

    List<Certificate> caList = loadCertificates(ROOT_CA_CERT);
    assertEquals(1, caList.size());
    Certificate signerCert = caList.get(0);

    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(Files.toByteArray(new File(TEST_PRE_SCT))));

    LogSignatureVerifier verifier = getVerifier();
    assertTrue("Expected signature to verify OK",
        verifier.verifySCTOverPreCertificate(sct, (X509Certificate) preCertificate,
            LogSignatureVerifier.issuerInformationFromCertificateIssuer(signerCert)));
  }

  /**
   * Tests for the public verifySignature method taking a chain of certificates.
   */
  @Test
  public void signatureOnRegularCertChainVerifies() throws IOException {
    // Flow:
    // test-cert.pem -> ca-cert.pem
    List<Certificate> certs = loadCertificates(TEST_CERT);
    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(Files.toByteArray(new File(TEST_CERT_SCT))));

    assertTrue(getVerifier().verifySignature(sct, certs));
  }

  @Test
  public void signatureOnCertSignedByIntermediateVerifies() throws IOException {
    // Flow:
    // test-intermediate-cert.pem -> intermediate-cert.pem -> ca-cert.pem
    List<Certificate> certsChain = new ArrayList<Certificate>();
    certsChain.addAll(loadCertificates(TEST_INTERMEDIATE_CERT));
    certsChain.addAll(loadCertificates(INTERMEDIATE_CA_CERT));
    certsChain.addAll(loadCertificates(ROOT_CA_CERT));
    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(Files.toByteArray(new File(TEST_INTERMEDIATE_CERT_SCT))));

    assertTrue(getVerifier().verifySignature(sct, certsChain));
  }


  @Test
  public void signatureOnPreCertificateCertsChainVerifies() throws IOException {
    // Flow:
    // test-embedded-pre-cert.pem -> ca-cert.pem
    List<Certificate> certsChain = new ArrayList<Certificate>();
    certsChain.addAll(loadCertificates(TEST_PRE_CERT));
    certsChain.addAll(loadCertificates(ROOT_CA_CERT));

    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(Files.toByteArray(new File(TEST_PRE_SCT))));

    assertTrue(getVerifier().verifySignature(sct, certsChain));
  }

  @Test
  public void signatureOnPreCertificateSignedByPreCertificateSigningCertVerifies()
      throws IOException {
    // Flow:
    // test-embedded-with-preca-pre-cert.pem -> ca-pre-cert.pem -> ca-cert.pem
    List<Certificate> certsChain = new ArrayList<Certificate>();
    certsChain.addAll(loadCertificates(TEST_PRE_CERT_SIGNED_BY_PRECA_CERT));
    certsChain.addAll(loadCertificates(PRE_CERT_SIGNING_CERT));
    certsChain.addAll(loadCertificates(ROOT_CA_CERT));

    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(Files.toByteArray(new File(TEST_PRE_CERT_PRECA_SCT))));

    assertTrue("Expected PreCertificate to verify OK",
        getVerifier().verifySignature(sct, certsChain));
  }

  @Test
  public void signatureOnPreCertificateSignedByIntermediateVerifies()
      throws IOException {
    // Flow:
    // test-embedded-with-intermediate-cert.pem -> intermediate-cert.pem -> ca-cert.pem
    List<Certificate> certsChain = new ArrayList<Certificate>();
    certsChain.addAll(loadCertificates(TEST_PRE_CERT_SIGNED_BY_INTERMEDIATE));
    certsChain.addAll(loadCertificates(INTERMEDIATE_CA_CERT));
    certsChain.addAll(loadCertificates(ROOT_CA_CERT));

    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(
            Files.toByteArray(new File(TEST_PRE_CERT_SIGNED_BY_INTERMEDIATE_SCT))));

    assertTrue("Expected PreCertificate to verify OK",
        getVerifier().verifySignature(sct, certsChain));
  }

  @Test
  public void signatureOnPreCertificateSignedByPreCertSigningCertSignedByIntermediateVerifies()
      throws IOException {
    // Flow:
    // test-embedded-with-intermediate-preca-pre-cert.pem -> intermediate-pre-cert.pem
    //   -> intermediate-cert.pem -> ca-cert.pem
    List<Certificate> certsChain = new ArrayList<Certificate>();
    certsChain.addAll(loadCertificates(TEST_PRE_CERT_SIGNED_BY_PRECA_INTERMEDIATE));
    certsChain.addAll(loadCertificates(PRE_CERT_SIGNING_BY_INTERMEDIATE));
    certsChain.addAll(loadCertificates(INTERMEDIATE_CA_CERT));
    certsChain.addAll(loadCertificates(ROOT_CA_CERT));

    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(
            Files.toByteArray(new File(TEST_PRE_CERT_SIGNED_BY_PRECA_INTERMEDIATE_SCT))));

    assertTrue("Expected PreCertificate to verify OK",
        getVerifier().verifySignature(sct, certsChain));
  }

  @Test
  public void throwsWhenChainWithPreCertificateSignedByPreCertificateSigningCertMissingIssuer()
      throws IOException {
    List<Certificate> certsChain = new ArrayList<Certificate>();
    certsChain.addAll(loadCertificates(TEST_PRE_CERT_SIGNED_BY_PRECA_CERT));
    certsChain.addAll(loadCertificates(PRE_CERT_SIGNING_CERT));

    Ct.SignedCertificateTimestamp sct = Deserializer.parseSCTFromBinary(
        new ByteArrayInputStream(Files.toByteArray(new File(TEST_PRE_CERT_PRECA_SCT))));

    try {
      getVerifier().verifySignature(sct, certsChain);
      fail("Expected verifySignature to throw since the issuer certificate is missing.");
    } catch (IllegalArgumentException expected) {
      assertNotNull("Exception should have message, but was: " + expected, expected.getMessage());
      assertTrue("Expected exception to warn about missing issuer cert",
          expected.getMessage().contains("must contain issuer"));
    }
  }
}
