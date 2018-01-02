package org.certificatetransparency.ctlog.serialization;

import com.google.common.base.Preconditions;
import com.google.protobuf.ByteString;

import org.apache.commons.codec.binary.Base64;
import org.certificatetransparency.ctlog.ParsedLogEntry;
import org.certificatetransparency.ctlog.ParsedLogEntryWithProof;
import org.certificatetransparency.ctlog.proto.Ct;
import org.json.simple.JSONArray;

import java.io.IOException;
import java.io.InputStream;

/**
 * Converting binary data to CT structures.
 */
public class Deserializer {
  /**
   * Parses a SignedCertificateTimestamp from binary encoding.
   * @param inputStream byte stream of binary encoding.
   * @return Built CT.SignedCertificateTimestamp
   * @throws SerializationException if the data stream is too short.
   */
  public static Ct.SignedCertificateTimestamp parseSCTFromBinary(InputStream inputStream) {
    Ct.SignedCertificateTimestamp.Builder sctBuilder = Ct.SignedCertificateTimestamp.newBuilder();

    int version = (int) readNumber(inputStream, 1 /* single byte */);
    if (version != Ct.Version.V1.getNumber()) {
      throw new SerializationException(String.format("Unknown version: %d", version));
    }
    sctBuilder.setVersion(Ct.Version.valueOf(version));

    byte[] keyId = readFixedLength(inputStream, CTConstants.KEY_ID_LENGTH);
    sctBuilder.setId(Ct.LogID.newBuilder().setKeyId(ByteString.copyFrom(keyId)).build());

    long timestamp = readNumber(inputStream, CTConstants.TIMESTAMP_LENGTH);
    sctBuilder.setTimestamp(timestamp);

    byte[] extensions = readVariableLength(inputStream, CTConstants.MAX_EXTENSIONS_LENGTH);
    sctBuilder.setExtensions(ByteString.copyFrom(extensions));

    sctBuilder.setSignature(parseDigitallySignedFromBinary(inputStream));
    return sctBuilder.build();
  }

  /**
   * Parses a Ct.DigitallySigned from binary encoding.
   * @param inputStream byte stream of binary encoding.
   * @return Built Ct.DigitallySigned
   * @throws SerializationException if the data stream is too short.
   */
  public static Ct.DigitallySigned parseDigitallySignedFromBinary(InputStream inputStream) {
    Ct.DigitallySigned.Builder builder = Ct.DigitallySigned.newBuilder();
    int hashAlgorithmByte = (int) readNumber(inputStream, 1 /* single byte */);
    Ct.DigitallySigned.HashAlgorithm hashAlgorithm =
        Ct.DigitallySigned.HashAlgorithm.valueOf(hashAlgorithmByte);
    if (hashAlgorithm == null) {
      throw new SerializationException(
          String.format("Unknown hash algorithm: %x", hashAlgorithmByte));
    }
    builder.setHashAlgorithm(hashAlgorithm);

    int signatureAlgorithmByte = (int) readNumber(inputStream, 1 /* single byte */);
    Ct.DigitallySigned.SignatureAlgorithm signatureAlgorithm =
        Ct.DigitallySigned.SignatureAlgorithm.valueOf(signatureAlgorithmByte);
    if (signatureAlgorithm == null) {
      throw new SerializationException(
          String.format("Unknown signature algorithm: %x", signatureAlgorithmByte));
    }
    builder.setSigAlgorithm(signatureAlgorithm);

    byte[] signature = readVariableLength(inputStream, CTConstants.MAX_SIGNATURE_LENGTH);
    builder.setSignature(ByteString.copyFrom(signature));

    return builder.build();
  }

  /**
   * Parses an entry retrieved from Log and it's audit proof.
   * @param entry ParsedLogEntry instance.
   * @param proof An array of base64-encoded Merkle Tree nodes proving the inclusion of the
   * chosen certificate.
   * @param leafIndex The index of the desired entry.
   * @param treeSize The tree size of the tree for which the proof is desired.
   * @return {@link ParsedLogEntryWithProof}
   */
  public static ParsedLogEntryWithProof parseLogEntryWithProof(ParsedLogEntry entry,
    JSONArray proof, long leafIndex, long treeSize) {

    Ct.MerkleAuditProof.Builder proofBuilder = Ct.MerkleAuditProof.newBuilder();
    proofBuilder.setVersion(Ct.Version.V1);
    proofBuilder.setLeafIndex(leafIndex);
    proofBuilder.setTreeSize(treeSize);

    for (Object node: proof) {
      proofBuilder.addPathNode(ByteString.copyFrom(Base64.decodeBase64((String) node)));
    }
    return ParsedLogEntryWithProof.newInstance(entry, proofBuilder.build());
  }

  /**
   * Parses an entry retrieved from Log.
   * @param merkleTreeLeaf MerkleTreeLeaf structure, byte stream of binary encoding.
   * @param extraData extra data,  byte stream of binary encoding.
   * @return {@link ParsedLogEntry}
   */
  public static ParsedLogEntry parseLogEntry(InputStream merkleTreeLeaf, InputStream extraData) {
    Ct.MerkleTreeLeaf treeLeaf = parseMerkleTreeLeaf(merkleTreeLeaf);
    Ct.LogEntry.Builder logEntryBuilder = Ct.LogEntry.newBuilder();

    Ct.LogEntryType entryType = treeLeaf.getTimestampedEntry().getEntryType();

    if (entryType == Ct.LogEntryType.X509_ENTRY) {
      Ct.X509ChainEntry x509EntryChain = parseX509ChainEntry(extraData,
        treeLeaf.getTimestampedEntry().getSignedEntry().getX509());
      logEntryBuilder.setX509Entry(x509EntryChain);

    } else if (entryType == Ct.LogEntryType.PRECERT_ENTRY) {
      Ct.PrecertChainEntry preCertChain = parsePrecerChainEntry(extraData,
        treeLeaf.getTimestampedEntry().getSignedEntry().getPrecert());
       logEntryBuilder.setPrecertEntry(preCertChain);

    } else {
      throw new SerializationException(String.format("Unknown entry type: %d", entryType));
    }
    Ct.LogEntry logEntry = logEntryBuilder.build();
    return ParsedLogEntry.newInstance(treeLeaf, logEntry);
  }

  /**
   * Parses a {@link Ct.MerkleTreeLeaf} from binary encoding.
   * @param in byte stream of binary encoding.
   * @return Built {@link Ct.MerkleTreeLeaf}.
   * @throws SerializationException if the data stream is too short.
   */
  public static Ct.MerkleTreeLeaf parseMerkleTreeLeaf(InputStream in) {
    Ct.MerkleTreeLeaf.Builder merkleTreeLeafBuilder = Ct.MerkleTreeLeaf.newBuilder();

    int version = (int) readNumber(in, CTConstants.VERSION_LENGTH);
    if (version != Ct.Version.V1.getNumber()) {
      throw new SerializationException(String.format("Unknown version: %d", version));
    }
    merkleTreeLeafBuilder.setVersion(Ct.Version.valueOf(version));

    int leafType = (int) readNumber(in, 1);
    if (leafType != Ct.MerkleLeafType.TIMESTAMPED_ENTRY_VALUE) {
      throw new SerializationException(String.format("Unknown entry type: %d", leafType));
    }
    merkleTreeLeafBuilder.setType(Ct.MerkleLeafType.valueOf(leafType));
    merkleTreeLeafBuilder.setTimestampedEntry((parseTimestampedEntry(in)));

    return merkleTreeLeafBuilder.build();
  }

  /**
   * Parses a {@link Ct.TimestampedEntry} from binary encoding.
   * @param in byte stream of binary encoding.
   * @return Built {@link Ct.TimestampedEntry}.
   * @throws SerializationException if the data stream is too short.
   */
  public static Ct.TimestampedEntry parseTimestampedEntry(InputStream in) {
    Ct.TimestampedEntry.Builder timestampedEntry = Ct.TimestampedEntry.newBuilder();

    long timestamp = readNumber(in, CTConstants.TIMESTAMP_LENGTH);
    timestampedEntry.setTimestamp(timestamp);

    int entryType = (int) readNumber(in, CTConstants.LOG_ENTRY_TYPE_LENGTH);
    timestampedEntry.setEntryType(Ct.LogEntryType.valueOf(entryType));

    Ct.SignedEntry.Builder signedEntryBuilder = Ct.SignedEntry.newBuilder();
    if (entryType == Ct.LogEntryType.X509_ENTRY_VALUE) {

      int length = (int) readNumber(in, 3);
      ByteString x509 = ByteString.copyFrom(readFixedLength(in, length));
      signedEntryBuilder.setX509(x509);

    } else if (entryType == Ct.LogEntryType.PRECERT_ENTRY_VALUE) {
      Ct.PreCert.Builder preCertBuilder = Ct.PreCert.newBuilder();

      byte[] arr = readFixedLength(in, 32);
      preCertBuilder.setIssuerKeyHash(ByteString.copyFrom(arr));

      // set tbs certificate
      arr = readFixedLength(in, 2);
      int length = (int) readNumber(in, 2);

      preCertBuilder.setTbsCertificate(ByteString.copyFrom(readFixedLength(in, length)));
      preCertBuilder.build();

      signedEntryBuilder.setPrecert(preCertBuilder);
    } else {
      throw new SerializationException(String.format("Unknown entry type: %d", entryType));
    }
    signedEntryBuilder.build();
    timestampedEntry.setSignedEntry(signedEntryBuilder);

    return timestampedEntry.build();
  }

  /**
   * Parses X509ChainEntry structure.
   * @param in X509ChainEntry structure, byte stream of binary encoding.
   * @param x509Cert leaf certificate.
   * @throws SerializationException if an I/O error occurs.
   * @return {@link Ct.X509ChainEntry} proto object.
   */
  public static Ct.X509ChainEntry parseX509ChainEntry(InputStream in, ByteString x509Cert) {
    Ct.X509ChainEntry.Builder x509EntryChain = Ct.X509ChainEntry.newBuilder();
    x509EntryChain.setLeafCertificate(x509Cert);

    try {
      if (readNumber(in, 3) != in.available()) {
        throw new SerializationException("Extra data corrupted.");
      }
      while (in.available() > 0) {
        int length = (int) readNumber(in, 3);
        x509EntryChain.addCertificateChain(ByteString.copyFrom(readFixedLength(in, length)));
      }
    } catch (IOException e) {
      throw new SerializationException("Cannot parse xChainEntry. " + e.getLocalizedMessage());
    }

    return x509EntryChain.build();
  }

  /**
   * Parses PrecertChainEntry structure.
   * @param in PrecertChainEntry structure, byte stream of binary encoding.
   * @param preCert Precertificate.
   * @return {@link Ct.PrecertChainEntry} proto object.
   */
  public static Ct.PrecertChainEntry parsePrecerChainEntry(InputStream in, Ct.PreCert preCert) {
    Ct.PrecertChainEntry.Builder preCertChain = Ct.PrecertChainEntry.newBuilder();
    preCertChain.setPreCert(preCert);

    try {
      if (readNumber(in, 3) != in.available()) {
        throw new SerializationException("Extra data corrupted.");
      }
      while (in.available() > 0) {
        int length = (int) readNumber(in, 3);
        preCertChain.addPrecertificateChain(ByteString.copyFrom(readFixedLength(in, length)));
      }
    } catch (IOException e) {
      throw new SerializationException("Cannot parse PrecertEntryChain."
        + e.getLocalizedMessage());
    }
    return preCertChain.build();
  }

  /**
   * Reads a variable-length byte array with a maximum length.
   * The length is read (based on the number of bytes needed to represent the max data length)
   * then the byte array itself.
   * @param inputStream byte stream of binary encoding.
   * @param maxDataLength Maximal data length.
   * @return read byte array.
   * @throws SerializationException if the data stream is too short.
   */
  static byte[] readVariableLength(InputStream inputStream, int maxDataLength) {
    int bytesForDataLength = bytesForDataLength(maxDataLength);
    long dataLength = readNumber(inputStream, bytesForDataLength);

    byte[] rawData = new byte[(int) dataLength];
    int bytesRead;
    try {
      bytesRead = inputStream.read(rawData);
    } catch (IOException e) {
      //Note: A finer-grained exception type should be thrown if the client
      // ever cares to handle transient I/O errors.
      throw new SerializationException("Error while reading variable-length data", e);
    }

    if (bytesRead != dataLength) {
      throw new SerializationException(String.format("Incomplete data. Expected %d bytes, had %d.",
          dataLength, bytesRead));
    }

    return rawData;
  }

  /**
   * Reads a fixed-length byte array.
   * @param inputStream byte stream of binary encoding.
   * @param dataLength exact data length.
   * @return read byte array.
   * @throws SerializationException if the data stream is too short.
   */
  static byte[] readFixedLength(InputStream inputStream, int dataLength) {
    byte[] toReturn = new byte[dataLength];
    try {
      int bytesRead = inputStream.read(toReturn);
      if (bytesRead < dataLength) {
        throw new SerializationException(
            String.format("Not enough bytes: Expected %d, got %d.", dataLength, bytesRead));
      }
      return toReturn;
    } catch (IOException e) {
      throw new SerializationException("Error while reading fixed-length buffer", e);
    }
  }

  /**
   * Calculates the number of bytes needed to hold the given number:
   * ceil(log2(maxDataLength)) / 8
   * @param maxDataLength
   * @return
   */
  public static int bytesForDataLength(int maxDataLength) {
    return (int) (Math.ceil(Math.log(maxDataLength) / Math.log(2)) / 8);
  }

  /**
   * Read a number of numBytes bytes (Assuming MSB first).
   * @param inputStream byte stream of binary encoding.
   * @param numBytes exact number of bytes representing this number.
   * @return a number of at most 2^numBytes
   */
  static long readNumber(InputStream inputStream, int numBytes) {
    Preconditions.checkArgument(numBytes <= 8, "Could not read a number of more than 8 bytes.");

    long toReturn = 0;
    try {
      for (int i = 0; i < numBytes; i++) {
        int valRead = inputStream.read();
        if (valRead < 0) {
          throw new SerializationException(
              String.format("Missing length bytes: Expected %d, got %d.", numBytes, i));
        }
        toReturn = (toReturn <<  8) | valRead;
      }
      return toReturn;
    } catch (IOException e) {
      throw new SerializationException("IO Error when reading number", e);
    }
  }
}
