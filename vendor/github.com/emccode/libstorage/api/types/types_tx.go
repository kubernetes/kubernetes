package types

import (
	"fmt"
	"regexp"
	"strconv"
	"time"

	"github.com/akutz/goof"
)

// TxTimestamp is a transaction's timestamp.
type TxTimestamp time.Time

// Transaction contains transaction information.
type Transaction struct {

	// ID is the transaction's ID.
	ID *UUID `json:"id" yaml:"id"`

	// Created is the UTC timestampe at which the transaction was created.
	Created TxTimestamp `json:"created"`
}

// NewTransaction returns a new transaction.
func NewTransaction() (*Transaction, error) {
	txID, err := NewUUID()
	if err != nil {
		return nil, err
	}
	return &Transaction{
		ID:      txID,
		Created: TxTimestamp(time.Now().UTC()),
	}, nil
}

// String returns the string representation of the transaction.
func (t *Transaction) String() string {
	return fmt.Sprintf("txID=%s, txCR=%s", t.ID, t.Created)
}

// MarshalText marshals the Transaction to a string.
func (t *Transaction) MarshalText() ([]byte, error) {
	return []byte(t.String()), nil
}

var txRX = regexp.MustCompile(`(?i)^txID=(.+),\s*txCR=(\d+)$`)

// UnmarshalText unmarshals the Transaction from a string.
func (t *Transaction) UnmarshalText(text []byte) error {

	m := txRX.FindSubmatch(text)
	if len(m) == 0 {
		return goof.WithField("value", string(text), "invalid transaction")
	}

	t.ID = &UUID{}
	if err := t.ID.UnmarshalText(m[1]); err != nil {
		return err
	}

	if err := (&t.Created).UnmarshalText(m[2]); err != nil {
		return err
	}

	return nil
}

// String returns the timestamp's epoch.
func (t TxTimestamp) String() string {
	return fmt.Sprintf("%d", time.Time(t).Unix())
}

// MarshalText marshals the TxTimestamp to a string.
func (t TxTimestamp) MarshalText() ([]byte, error) {
	return []byte(t.String()), nil
}

// UnmarshalText unmarshals the timestamp from a string to a TxTimestamp.
func (t *TxTimestamp) UnmarshalText(text []byte) error {
	i, err := strconv.ParseInt(string(text), 10, 64)
	if err != nil {
		return err
	}
	*t = TxTimestamp(time.Unix(i, 0))
	return nil
}

// ContextLoggerFields indicate to the context logger what data to log.
func (t *Transaction) ContextLoggerFields() map[string]interface{} {
	return map[string]interface{}{
		"txID": t.ID.String(),
		"txCR": t.Created.String(),
	}
}
