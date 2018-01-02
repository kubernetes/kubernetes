package org.certificatetransparency.ctlog;

import org.certificatetransparency.ctlog.proto.Ct;

/**
 * ParsedLogEntry data type contains an entry retrieved from Log with it's audit proof.
 */
public class ParsedLogEntryWithProof  {
  private final ParsedLogEntry parsedLogEntry;
  private final Ct.MerkleAuditProof auditProof;

  private ParsedLogEntryWithProof(ParsedLogEntry parsedLogEntry, Ct.MerkleAuditProof auditProof) {
    this.parsedLogEntry = parsedLogEntry;
    this.auditProof = auditProof;
  }

  public static ParsedLogEntryWithProof newInstance(ParsedLogEntry logEntry,
    Ct.MerkleAuditProof proof) {
    return new ParsedLogEntryWithProof(logEntry, proof);
  }

  public ParsedLogEntry getParsedLogEntry() {
    return parsedLogEntry;
  }

  public Ct.MerkleAuditProof getAuditProof() {
    return auditProof;
  }
}
