package org.certificatetransparency.ctlog.serialization;

import com.google.common.io.Files;
import com.google.protobuf.ByteString;

import org.apache.commons.codec.binary.Base64;
import org.certificatetransparency.ctlog.proto.Ct;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.io.IOException;

/**
 * Test serialization.
 */
@RunWith(JUnit4.class)
public class TestSerializer {
  public static final String TEST_CERT_SCT = "test/testdata/test-cert.proof";

  @Test
  public void serializeSCT() throws IOException {
    Ct.SignedCertificateTimestamp.Builder builder = Ct.SignedCertificateTimestamp.newBuilder();
    builder.setVersion(Ct.Version.V1);
    builder.setTimestamp(1365181456089L);

    String keyIdBase64 = "3xwuwRUAlFJHqWFoMl3cXHlZ6PfG04j8AC4LvT9012Q=";
    builder.setId(Ct.LogID.newBuilder().setKeyId(
        ByteString.copyFrom(Base64.decodeBase64(keyIdBase64))).build());

    String signatureBase64 =
        "MEUCIGBuEK5cLVobCu1J3Ek39I3nGk6XhOnCCN+/6e9TbPfy" +
        "AiEAvrKcctfQbWHQa9s4oGlGmqhv4S4Yu3zEVomiwBh+9aU=";

    Ct.DigitallySigned.Builder signatureBuilder = Ct.DigitallySigned.newBuilder();
    signatureBuilder.setHashAlgorithm(Ct.DigitallySigned.HashAlgorithm.SHA256);
    signatureBuilder.setSigAlgorithm(Ct.DigitallySigned.SignatureAlgorithm.ECDSA);
    signatureBuilder.setSignature(ByteString.copyFrom(Base64.decodeBase64(signatureBase64)));

    builder.setSignature(signatureBuilder.build());

    byte[] generatedBytes = Serializer.serializeSctToBinary(builder.build());
    byte[] readBytes = Files.toByteArray(new File(TEST_CERT_SCT));
    Assert.assertArrayEquals(readBytes, generatedBytes);
  }
}
