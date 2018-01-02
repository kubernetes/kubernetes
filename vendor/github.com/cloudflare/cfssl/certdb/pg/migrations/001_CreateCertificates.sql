-- +goose Up
-- SQL in section 'Up' is executed when this migration is applied

CREATE TABLE certificates (
  serial_number            bytea NOT NULL,
  authority_key_identifier bytea NOT NULL,
  ca_label                 bytea,
  status                   bytea NOT NULL,
  reason                   int,
  expiry                   timestamptz,
  revoked_at               timestamptz,
  pem                      bytea NOT NULL,
  PRIMARY KEY(serial_number, authority_key_identifier)
);

CREATE TABLE ocsp_responses (
  serial_number            bytea NOT NULL,
  authority_key_identifier bytea NOT NULL,
  body                     bytea NOT NULL,
  expiry                   timestamptz,
  PRIMARY KEY(serial_number, authority_key_identifier),
  FOREIGN KEY(serial_number, authority_key_identifier) REFERENCES certificates(serial_number, authority_key_identifier)
);
-- +goose Down
-- SQL section 'Down' is executed when this migration is rolled back

DROP TABLE ocsp_responses;
DROP TABLE certificates;
