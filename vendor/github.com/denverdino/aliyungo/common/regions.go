package common

// Region represents ECS region
type Region string

// Constants of region definition
const (
	Hangzhou     = Region("cn-hangzhou")
	Qingdao      = Region("cn-qingdao")
	Beijing      = Region("cn-beijing")
	Hongkong     = Region("cn-hongkong")
	Shenzhen     = Region("cn-shenzhen")
	USWest1      = Region("us-west-1")
	USEast1      = Region("us-east-1")
	APSouthEast1 = Region("ap-southeast-1")
	Shanghai     = Region("cn-shanghai")
	MEEast1      = Region("me-east-1")
	APNorthEast1 = Region("ap-northeast-1")
	APSouthEast2 = Region("ap-southeast-2")
	EUCentral1   = Region("eu-central-1")
)

var ValidRegions = []Region{
	Hangzhou, Qingdao, Beijing, Shenzhen, Hongkong, Shanghai,
	USWest1, USEast1,
	APNorthEast1, APSouthEast1, APSouthEast2,
	MEEast1,
	EUCentral1,
}
