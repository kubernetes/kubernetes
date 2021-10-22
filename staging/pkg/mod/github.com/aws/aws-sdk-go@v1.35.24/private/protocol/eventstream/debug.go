package eventstream

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
)

type decodedMessage struct {
	rawMessage
	Headers decodedHeaders `json:"headers"`
}
type jsonMessage struct {
	Length     json.Number    `json:"total_length"`
	HeadersLen json.Number    `json:"headers_length"`
	PreludeCRC json.Number    `json:"prelude_crc"`
	Headers    decodedHeaders `json:"headers"`
	Payload    []byte         `json:"payload"`
	CRC        json.Number    `json:"message_crc"`
}

func (d *decodedMessage) UnmarshalJSON(b []byte) (err error) {
	var jsonMsg jsonMessage
	if err = json.Unmarshal(b, &jsonMsg); err != nil {
		return err
	}

	d.Length, err = numAsUint32(jsonMsg.Length)
	if err != nil {
		return err
	}
	d.HeadersLen, err = numAsUint32(jsonMsg.HeadersLen)
	if err != nil {
		return err
	}
	d.PreludeCRC, err = numAsUint32(jsonMsg.PreludeCRC)
	if err != nil {
		return err
	}
	d.Headers = jsonMsg.Headers
	d.Payload = jsonMsg.Payload
	d.CRC, err = numAsUint32(jsonMsg.CRC)
	if err != nil {
		return err
	}

	return nil
}

func (d *decodedMessage) MarshalJSON() ([]byte, error) {
	jsonMsg := jsonMessage{
		Length:     json.Number(strconv.Itoa(int(d.Length))),
		HeadersLen: json.Number(strconv.Itoa(int(d.HeadersLen))),
		PreludeCRC: json.Number(strconv.Itoa(int(d.PreludeCRC))),
		Headers:    d.Headers,
		Payload:    d.Payload,
		CRC:        json.Number(strconv.Itoa(int(d.CRC))),
	}

	return json.Marshal(jsonMsg)
}

func numAsUint32(n json.Number) (uint32, error) {
	v, err := n.Int64()
	if err != nil {
		return 0, fmt.Errorf("failed to get int64 json number, %v", err)
	}

	return uint32(v), nil
}

func (d decodedMessage) Message() Message {
	return Message{
		Headers: Headers(d.Headers),
		Payload: d.Payload,
	}
}

type decodedHeaders Headers

func (hs *decodedHeaders) UnmarshalJSON(b []byte) error {
	var jsonHeaders []struct {
		Name  string      `json:"name"`
		Type  valueType   `json:"type"`
		Value interface{} `json:"value"`
	}

	decoder := json.NewDecoder(bytes.NewReader(b))
	decoder.UseNumber()
	if err := decoder.Decode(&jsonHeaders); err != nil {
		return err
	}

	var headers Headers
	for _, h := range jsonHeaders {
		value, err := valueFromType(h.Type, h.Value)
		if err != nil {
			return err
		}
		headers.Set(h.Name, value)
	}
	*hs = decodedHeaders(headers)

	return nil
}

func valueFromType(typ valueType, val interface{}) (Value, error) {
	switch typ {
	case trueValueType:
		return BoolValue(true), nil
	case falseValueType:
		return BoolValue(false), nil
	case int8ValueType:
		v, err := val.(json.Number).Int64()
		return Int8Value(int8(v)), err
	case int16ValueType:
		v, err := val.(json.Number).Int64()
		return Int16Value(int16(v)), err
	case int32ValueType:
		v, err := val.(json.Number).Int64()
		return Int32Value(int32(v)), err
	case int64ValueType:
		v, err := val.(json.Number).Int64()
		return Int64Value(v), err
	case bytesValueType:
		v, err := base64.StdEncoding.DecodeString(val.(string))
		return BytesValue(v), err
	case stringValueType:
		v, err := base64.StdEncoding.DecodeString(val.(string))
		return StringValue(string(v)), err
	case timestampValueType:
		v, err := val.(json.Number).Int64()
		return TimestampValue(timeFromEpochMilli(v)), err
	case uuidValueType:
		v, err := base64.StdEncoding.DecodeString(val.(string))
		var tv UUIDValue
		copy(tv[:], v)
		return tv, err
	default:
		panic(fmt.Sprintf("unknown type, %s, %T", typ.String(), val))
	}
}
