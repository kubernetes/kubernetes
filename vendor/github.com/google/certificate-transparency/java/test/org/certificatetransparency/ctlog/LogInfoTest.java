package org.certificatetransparency.ctlog;

import org.apache.commons.codec.binary.Base64;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.security.KeyFactory;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.X509EncodedKeySpec;

/**
 * Mostly for verifying the log info calculates the log ID correctly.
 */
@RunWith(JUnit4.class)
public class LogInfoTest {
  public static final byte[] PUBLIC_KEY = Base64.decodeBase64(
      "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEfahLEimAoz2t01p3uMziiLOl/fHTDM0YDOhBRuiBARsV"
          + "4UvxG2LdNgoIGLrtCzWE0J5APC2em4JlvR8EEEFMoA==");
  public static final byte[] LOG_ID =
      Base64.decodeBase64("pLkJkLQYWBSHuxOizGdwCjw1mAT5G9+443fNDsgN3BA=");

  static PublicKey getKey() {
    X509EncodedKeySpec spec = new X509EncodedKeySpec(PUBLIC_KEY);
    try {
      KeyFactory kf = KeyFactory.getInstance("EC");
      return kf.generatePublic(spec);
    } catch (InvalidKeySpecException e) {
      throw new RuntimeException(e);
    } catch (NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testCalculatesLogIdCorrectly() throws NoSuchAlgorithmException {
    LogInfo logInfo = new LogInfo(getKey());
    Assert.assertTrue(logInfo.isSameLogId(LOG_ID));
  }
}
