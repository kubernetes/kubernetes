package cmdproto

import (
	"fmt"
	"reflect"
	"strconv"
	"time"
	"unsafe"

	"github.com/golang/protobuf/descriptor"
	"github.com/golang/protobuf/proto"
	"github.com/spf13/cobra"

	cmdproto "k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd"
)

func extractFlagDetailFromMessage(i int, msg interface{}) (string, string, []string, string, cmdproto.FlagDetail_EXT, error) {
	_, md := descriptor.ForMessage(msg.(descriptor.Message))
	opts := md.Field[i].GetOptions()
	info, err := proto.GetExtension(opts, cmdproto.E_Info)
	if err != nil {
		return "", "", []string{""}, "", cmdproto.FlagDetail_NONE, err
	}
	return info.(*cmdproto.FlagDetail).GetName(), info.(*cmdproto.FlagDetail).GetShorthand(), info.(*cmdproto.FlagDetail).GetValue(), info.(*cmdproto.FlagDetail).GetUsage(), info.(*cmdproto.FlagDetail).GetExt(), nil
}

func FlagsSetup(cmd *cobra.Command, msg interface{}) {
	v := reflect.ValueOf(msg).Elem()

	for i := 0; i < v.NumField()-1; i++ {
		name, shorthand, defaultValue, usage, ext, err := extractFlagDetailFromMessage(i, msg)
		if err != nil {
			panic(err)
		}

		switch v.Field(i).Interface().(type) {
		case *bool:
			var tmpBool bool
			bVal, err := strconv.ParseBool(defaultValue[0])
			if err != nil {
				panic(err)
			}
			cmd.Flags().BoolVarP(&tmpBool, name, shorthand, bVal, usage)
			v.Field(i).Set(reflect.ValueOf(&tmpBool))
		case *int64:
			iVal, err := strconv.ParseInt(defaultValue[0], 0, 64)
			if err != nil {
				panic(err)
			}
			if ext == cmdproto.FlagDetail_ISTIME {
				var tmpTime time.Duration
				cmd.Flags().DurationVarP(&tmpTime, name, shorthand, time.Duration(iVal), usage)
				v.Field(i).Set(reflect.ValueOf((*int64)(unsafe.Pointer(&tmpTime))))
			} else {
				var tmpInt64 int64
				cmd.Flags().Int64VarP(&tmpInt64, name, shorthand, iVal, usage)
				v.Field(i).Set(reflect.ValueOf(&tmpInt64))
			}
		case *int32:
			var tmpInt32 int32
			iVal, err := strconv.ParseInt(defaultValue[0], 0, 32)
			if err != nil {
				panic(err)
			}
			cmd.Flags().Int32VarP(&tmpInt32, name, shorthand, int32(iVal), usage)
			v.Field(i).Set(reflect.ValueOf(&tmpInt32))
		case *string:
			var tmpString string
			cmd.Flags().StringVarP(&tmpString, name, shorthand, defaultValue[0], usage)
			v.Field(i).Set(reflect.ValueOf(&tmpString))
		case *cmdproto.Array:
			switch ext {
			case cmdproto.FlagDetail_ISSTRSLICE:
				var tmpSliceString cmdproto.Array
				cmd.Flags().StringSliceVarP(&tmpSliceString.Array, name, shorthand, defaultValue, usage)
				v.Field(i).Set(reflect.ValueOf(&tmpSliceString))
			case cmdproto.FlagDetail_ISSTRARRAY:
				var tmpStringArray cmdproto.Array
				cmd.Flags().StringArrayVarP(&tmpStringArray.Array, name, shorthand, defaultValue, usage)
				v.Field(i).Set(reflect.ValueOf(&tmpStringArray))
			default:
				panic(fmt.Errorf("haven't implement yet"))
			}
		default:
			panic(fmt.Errorf("haven't implement yet"))
		}
	}
}
