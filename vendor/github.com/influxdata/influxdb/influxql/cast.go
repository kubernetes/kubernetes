package influxql

func castToFloat(v interface{}) float64 {
	switch v := v.(type) {
	case float64:
		return v
	case int64:
		return float64(v)
	default:
		return float64(0)
	}
}

func castToInteger(v interface{}) int64 {
	switch v := v.(type) {
	case float64:
		return int64(v)
	case int64:
		return v
	default:
		return int64(0)
	}
}

func castToString(v interface{}) string {
	switch v := v.(type) {
	case string:
		return v
	default:
		return ""
	}
}

func castToBoolean(v interface{}) bool {
	switch v := v.(type) {
	case bool:
		return v
	default:
		return false
	}
}
