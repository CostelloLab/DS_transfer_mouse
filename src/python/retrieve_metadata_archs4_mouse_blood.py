import archs4py as a4
import pandas as pd

#path to file
file = "/projects/lgillenwater@xsede.org/DS_transfer_mouse/data/mouse/archs4/mouse_gene_v2.2.h5"

# get sample meta data
sample_meta = a4.meta.samples(file, ["GSM1444479","GSM2094897","GSM2094892","GSM1444481","GSM2094891","GSM1444474","GSM1444477","GSM2094896","GSM1444483","GSM1444471","GSM1444478","GSM1444472","GSM2094898","GSM1444473","GSM1444482","GSM1444484","GSM2094895","GSM2094893","GSM1444476","GSM1444470","GSM2094894","GSM1444480","GSM1444475","GSM1493054","GSM1593009","GSM1493053","GSM1593013","GSM1493049","GSM2111472","GSM2176513","GSM2176514","GSM1493050","GSM1493046","GSM1493052","GSM1593008","GSM2111473","GSM1493051","GSM1593014","GSM1593011","GSM1493048","GSM2176515","GSM1493055","GSM1493047","GSM1593012","GSM1593010","GSM1593007","GSM2154923","GSM2154931","GSM2154932","GSM2154906","GSM2154910","GSM2154936","GSM2154939","GSM2154907","GSM2154909","GSM2154911","GSM2154914","GSM2154917","GSM2154924","GSM2154925","GSM2154930","GSM2154948","GSM2154908","GSM2154929","GSM2154921","GSM2154915","GSM2154933","GSM2154940","GSM2154935","GSM2154913","GSM2154943","GSM2154937","GSM2154947","GSM2154952","GSM2154920","GSM2154919","GSM2154949","GSM2154945","GSM2154951","GSM2154938","GSM2154950","GSM2154941","GSM2154942","GSM2154953","GSM2154918","GSM2154922","GSM2154927","GSM2154946","GSM2154944","GSM2154916","GSM2154934","GSM2154928","GSM2154926","GSM2154912","GSM2590427","GSM2590428","GSM2590429","GSM2590430","GSM2590431","GSM2593919","GSM2593920","GSM2593921","GSM2593922","GSM2593923","GSM2593924","GSM2593925","GSM2593926","GSM2593927","GSM2593928","GSM2593929","GSM2593930","GSM2593931","GSM2593932","GSM2593933","GSM2593934","GSM2593935","GSM2593936","GSM2593937","GSM2593938","GSM2593939","GSM2593940","GSM2593941","GSM2593942","GSM2593943","GSM2593944","GSM2593945","GSM2593946","GSM2593947","GSM2593948","GSM2593949","GSM2593950","GSM2593951","GSM2593952","GSM2593953","GSM2593954","GSM2593955","GSM2593956","GSM2593957","GSM2593958","GSM2593959","GSM2593960","GSM2888960","GSM2888961","GSM2888962","GSM2888963","GSM2888964","GSM2888965","GSM2888966","GSM2888967","GSM2888968","GSM2888969","GSM2888970","GSM2888971","GSM2888972","GSM2888973","GSM2888974","GSM2888975","GSM2888976","GSM2888977","GSM2888978","GSM2888979","GSM2888980","GSM2888981","GSM1944768","GSM1944769","GSM1944770","GSM1944771","GSM1944772","GSM1944773","GSM1944774","GSM1944775","GSM1944776","GSM1944777","GSM1944778","GSM1944779","GSM1944780","GSM1944781","GSM1944782","GSM1944783","GSM1944784","GSM1944785","GSM1944786","GSM1944787","GSM1944788","GSM1944789","GSM1944790","GSM1944791","GSM2795210","GSM2906459","GSM2906460","GSM2906461","GSM2906462","GSM2906463","GSM2906464","GSM2947933","GSM2477962","GSM2477963","GSM2477964","GSM2477969","GSM2477970","GSM2477971","GSM2477972","GSM2477973","GSM2477974","GSM2510178","GSM2510179","GSM2510180","GSM2510181","GSM2510182","GSM2510183","GSM2510184","GSM2510185","GSM2510186","GSM2510187","GSM2510203","GSM2510204","GSM2510205","GSM2510206","GSM2510207","GSM2510208","GSM2510209","GSM2510210","GSM2510211","GSM2510212","GSM2510213","GSM2510214","GSM2510215","GSM2510216","GSM2510217","GSM2510218","GSM2510219","GSM2510220","GSM2510221","GSM2510222","GSM2510223","GSM2510224","GSM2757284","GSM2757285","GSM2757286","GSM2757287","GSM2757288","GSM2757289","GSM2757290","GSM2757291","GSM2757292","GSM2757293","GSM3048659","GSM3048660","GSM3141669","GSM3141670","GSM3141671","GSM3048657","GSM3048658","GSM2857709","GSM2857710","GSM2089172","GSM2089173","GSM2089174","GSM2730340","GSM2730341","GSM3377682","GSM3377683","GSM3377684","GSM2991047","GSM2991048","GSM2991049","GSM2991050","GSM2991051","GSM2991052","GSM2991053","GSM2991054","GSM2991055","GSM2991056","GSM2991057","GSM2991058","GSM2991059","GSM3323500","GSM2879324","GSM2879325","GSM2879326","GSM2879327","GSM2879328","GSM2879329","GSM3244442","GSM3244443","GSM3244444","GSM3244445","GSM3244446","GSM3244447","GSM3244448","GSM3244449","GSM3244450","GSM3244451","GSM3244452","GSM3244453","GSM3244454","GSM3244455","GSM3351706","GSM3351707","GSM3351708","GSM3675906","GSM3675907","GSM3675908","GSM3675909","GSM3675910","GSM3675911","GSM3675912","GSM3675913","GSM3132444","GSM3132445","GSM3132446","GSM3132447","GSM3132448","GSM3132449","GSM3132450","GSM3132451","GSM3132452","GSM3132453","GSM3132454","GSM3132455","GSM3212577","GSM3212582","GSM3212602","GSM3212603","GSM3576113","GSM3576114","GSM3576115","GSM3576116","GSM3576117","GSM3576118","GSM3576119","GSM3576120","GSM3576121","GSM3576122","GSM3576123","GSM3576124","GSM3576125","GSM3576126","GSM3576127","GSM3576128","GSM3385532","GSM3385533","GSM3385534","GSM3385535","GSM3385536","GSM3385537","GSM3385538","GSM3385539","GSM3385540","GSM3385541","GSM3385542","GSM3385543","GSM3385544","GSM3385545","GSM3385546","GSM3385547","GSM3385548","GSM3385549","GSM3385550","GSM3385551","GSM3385552","GSM3385553","GSM3385554","GSM3385555","GSM3385698","GSM3385699","GSM3385700","GSM3385701","GSM3385702","GSM3385703","GSM3385704","GSM3385705","GSM3385706","GSM3385738","GSM3385739","GSM3385740","GSM3385741","GSM3385742","GSM3385743","GSM3385744","GSM3385745","GSM3385746","GSM3385747","GSM3385748","GSM3385749","GSM3385750","GSM3385751","GSM3385752","GSM3385753","GSM3385754","GSM3385755","GSM3385756","GSM3385757","GSM3385758","GSM3385759","GSM3385760","GSM3385761","GSM3385762","GSM3385763","GSM3385764","GSM3385765","GSM3385766","GSM3385767","GSM3385768","GSM3828457","GSM3828458","GSM3828459","GSM3828460","GSM3828461","GSM3828462","GSM3828463","GSM3828464","GSM3828465","GSM3828466","GSM3828467","GSM3828468","GSM3828469","GSM3828470","GSM3828471","GSM3828472","GSM3828473","GSM3828474","GSM3828475","GSM3828476","GSM3828477","GSM3828478","GSM3828479","GSM3828480","GSM3828481","GSM3828482","GSM3828483","GSM3828484","GSM3828485","GSM3828486","GSM3828487","GSM3828488","GSM3828489","GSM3828490","GSM3828491","GSM3828492","GSM3828493","GSM3828494","GSM3828495","GSM3828496","GSM3828497","GSM3828498","GSM3828499","GSM3828500","GSM3828501","GSM3828502","GSM3828503","GSM3828504","GSM3828505","GSM3828506","GSM3828507","GSM3828508","GSM3828509","GSM3828510","GSM3828511","GSM3828512","GSM3828513","GSM3828514","GSM3828515","GSM3734716","GSM3734717","GSM2260320","GSM2260324","GSM2260325","GSM2260328","GSM2260330","GSM2260332","GSM2260334","GSM2260336","GSM2260338","GSM2260341","GSM3575967","GSM3575968","GSM3575969","GSM3575970","GSM3575971","GSM3575972","GSM3575973","GSM3575974","GSM3575975","GSM3575976","GSM3575977","GSM3575978","GSM3575979","GSM3575980","GSM3575981","GSM3575982","GSM3575983","GSM3575984","GSM3575985","GSM3575986","GSM3575987","GSM3575988","GSM3575989","GSM3025484","GSM3025485","GSM3025486","GSM3025487","GSM3025488","GSM3025489","GSM3025490","GSM3025491","GSM3025492","GSM3712194","GSM3712195","GSM3712196","GSM3712197",
"GSM3712198","GSM3712199","GSM3757822","GSM3757823","GSM3757824","GSM3757825","GSM3757826","GSM3757827","GSM3757829","GSM3757830","GSM3757832","GSM3757833","GSM3757834","GSM3757835","GSM3757836","GSM3757837","GSM3486788","GSM3486789","GSM3486790","GSM3486791","GSM3486792","GSM3486793","GSM3486794","GSM3486795","GSM3486796","GSM3486797","GSM3486798","GSM3486799","GSM3486800","GSM3486801",
"GSM3486802","GSM3486803","GSM3486804","GSM3486805","GSM3486806","GSM3486807","GSM3486808","GSM3486809","GSM3486810","GSM3486811","GSM3486812","GSM4119767","GSM4119768","GSM4119769","GSM4119770","GSM4119771","GSM4119772","GSM3173808","GSM3173809","GSM3173810","GSM3173811","GSM3173826","GSM3173827","GSM3173828","GSM3173829","GSM3272285","GSM3272286","GSM3272287","GSM3616029","GSM3616030",
"GSM3616032","GSM3616046","GSM3616047","GSM3616048","GSM3616049","GSM3616050","GSM3616051","GSM3616053","GSM3616054","GSM3616055","GSM3616056","GSM3616057","GSM3616058","GSM3616059","GSM3509270","GSM3509271","GSM3509272","GSM3509273","GSM3509274","GSM3509275","GSM3509276","GSM3509277","GSM3954983","GSM3954984","GSM3954985","GSM3954986","GSM3954987","GSM3954988","GSM3954989","GSM3954990",
"GSM3954991","GSM3954992","GSM3954993","GSM3954994","GSM3954995","GSM3954996","GSM3954997","GSM3954998","GSM3954999","GSM3955000","GSM3955001","GSM3955002","GSM3955003","GSM3955004","GSM3955005","GSM3955006","GSM3955007","GSM3955008","GSM3955009","GSM3955010","GSM3955011","GSM3955012","GSM3955013","GSM3955014","GSM3955015","GSM3955016","GSM3955017","GSM3955018","GSM3955019","GSM3955020",
"GSM4114594","GSM4114595","GSM4114596","GSM4114597","GSM3892327","GSM3892328","GSM3892329","GSM3892330","GSM3892331","GSM3892332","GSM3892333","GSM3892334","GSM3892335","GSM3892336","GSM3892337","GSM2307480","GSM2307481","GSM2307482","GSM2307483","GSM2307484","GSM2307485","GSM2307486","GSM2307487","GSM2307488","GSM2307489","GSM2307490","GSM2307491","GSM2307492","GSM2307493","GSM2975849",
"GSM2975852","GSM2975863","GSM2975867","GSM2975893","GSM2975894","GSM2975898","GSM2975904","GSM2975913","GSM2975916","GSM2975928","GSM2975946","GSM2976007","GSM2976043","GSM2976048","GSM2976059","GSM2976062","GSM2976071","GSM2976076","GSM2976078","GSM2976083","GSM2976085","GSM2976090","GSM2976091","GSM2976094","GSM2976099","GSM2976104","GSM2976105","GSM2976108","GSM2976113","GSM2976122",
"GSM2976124","GSM2976126","GSM2976135","GSM2976137","GSM2976141","GSM2976396","GSM2976397","GSM2976398","GSM2976399","GSM3564834","GSM3564837","GSM3597104","GSM3597105","GSM3597106","GSM3597107","GSM3597108","GSM3597109","GSM3597110","GSM3597111","GSM3597112","GSM3597113","GSM3640082","GSM3640083","GSM3640084","GSM3640085","GSM3640086","GSM3640087","GSM3735194","GSM3735195","GSM3735196",
"GSM3735197","GSM3735198","GSM3735199","GSM3735200","GSM3735201","GSM3735202","GSM3735203","GSM3735204","GSM3735205","GSM3755560","GSM3891044","GSM3891045","GSM3891046","GSM3891047","GSM3891048","GSM3891049","GSM3891050","GSM3891051","GSM3891052","GSM3891053","GSM3891054","GSM3891055","GSM3891056","GSM3891057","GSM3891058","GSM3891059","GSM3891060","GSM3891061","GSM3891062","GSM3891063",
"GSM3891064","GSM3891065","GSM3891067","GSM3891068","GSM3891069","GSM3891070","GSM3891071","GSM3891072","GSM3891073","GSM3891074","GSM3891075","GSM3891076","GSM3891077","GSM3891078","GSM3891079","GSM3891080","GSM3891081","GSM3891082","GSM3891083","GSM3891084","GSM3891085","GSM3891086","GSM3891087","GSM3891088","GSM3891089","GSM3891090","GSM3891091","GSM3891092","GSM3891093","GSM3891094",
"GSM3891095","GSM3891096","GSM3891097","GSM3891098","GSM3891099","GSM3891100","GSM3891101","GSM3891102","GSM3891103","GSM3891104","GSM3891105","GSM3891106","GSM3891107","GSM3891108","GSM3891109","GSM3891110","GSM3891111","GSM3891112","GSM3891113","GSM3891114","GSM3891115","GSM3891116","GSM3891117","GSM3891118","GSM3891119","GSM3891120","GSM3891121","GSM3891122","GSM3891123","GSM3891124",
"GSM3891125","GSM3891126","GSM3891127","GSM3891128","GSM3891129","GSM3891130","GSM3891131","GSM3891132","GSM3891133","GSM3891134","GSM3891135","GSM3932569","GSM3932570","GSM3932571","GSM3932572","GSM3932573","GSM3932574","GSM3957928","GSM3957929","GSM3957930","GSM3957931","GSM3957932","GSM3957933","GSM3957934","GSM3957935","GSM3957936","GSM3957937","GSM3957938","GSM4059582","GSM4059583",
"GSM4059584","GSM4059585","GSM4059586","GSM4059587","GSM4059588","GSM4059589","GSM4059590","GSM4059591","GSM4059592","GSM4059593","GSM4059594","GSM4059595","GSM4059596","GSM4059597","GSM4059598","GSM4059599","GSM4059600","GSM4059601","GSM4059602","GSM4059603","GSM4059604","GSM4059605","GSM4059606","GSM4059607","GSM4059608","GSM4059609","GSM4059610","GSM4059611","GSM4059612","GSM4059613",
"GSM4066715","GSM4066716","GSM4066717","GSM4066718","GSM4066719","GSM4066720","GSM4066721","GSM4066722","GSM4066723","GSM4066724","GSM4066725","GSM4066726","GSM4066727","GSM4066728","GSM4066729","GSM4066730","GSM4066731","GSM4066732","GSM4066733","GSM4066734","GSM4066735","GSM4066736","GSM4066737","GSM4066738","GSM4066739","GSM4066740","GSM4066741","GSM4066742","GSM4066743","GSM4066744",
"GSM4066745","GSM4066746","GSM4066747","GSM4066748","GSM4066749","GSM4066750","GSM4066751","GSM4066752","GSM4066753","GSM4066754","GSM4066755","GSM4066756","GSM4066757","GSM4066758","GSM4066759","GSM4066760","GSM4066761","GSM4066762","GSM4066763","GSM4066764","GSM4066765","GSM4066766","GSM4066767","GSM4066768","GSM4066769","GSM4066770","GSM4066771","GSM4066772","GSM4190246","GSM4190247",
"GSM4190271","GSM4190546","GSM4190547","GSM4190548","GSM4190550","GSM4190551","GSM4190552","GSM4190553","GSM4190554","GSM4190555","GSM4190556","GSM4190557","GSM4190558","GSM4190559","GSM4190561","GSM4190562","GSM4190563","GSM4190564","GSM4190565","GSM4190566","GSM4190567","GSM4190568","GSM4190569","GSM4190570","GSM4190571","GSM4190572","GSM4190573","GSM4190574","GSM4190575","GSM4195028",
"GSM4195029","GSM4195030","GSM4195031","GSM4195032","GSM4196828","GSM4196829","GSM4196830","GSM4196831","GSM4196832","GSM4196833","GSM4196834","GSM4196835","GSM4196836","GSM4196837","GSM4196838","GSM4196839","GSM4196840","GSM4196841","GSM4196842","GSM4196843","GSM4196844","GSM4196845","GSM4196846","GSM4212689","GSM4212690","GSM4212691","GSM4227302","GSM4227303","GSM4227304","GSM4227305",
"GSM4239563","GSM4239564","GSM4239565","GSM4255677","GSM4255678","GSM4255679","GSM4255680","GSM4286781","GSM4286782","GSM4286783","GSM4286784","GSM4286785","GSM4286786","GSM4286787","GSM4286788","GSM4286789","GSM4286790","GSM4286791","GSM4286792","GSM4286793","GSM4286794","GSM4286795","GSM4286797","GSM4286798","GSM4286799","GSM4286800","GSM4286801","GSM4286802","GSM4286803","GSM4286804",
"GSM4286805","GSM4286806","GSM4286807","GSM4286808","GSM4286809","GSM4286810","GSM4286811","GSM4286812","GSM4286813","GSM4286814","GSM4286815","GSM4286816","GSM4286817","GSM4286818","GSM4286819","GSM4286820","GSM4286821","GSM4286822","GSM4286823","GSM4286824","GSM4286825","GSM4286826","GSM4286827","GSM4286828","GSM4286829","GSM4286830","GSM4286831","GSM4286832","GSM4286833","GSM4286834",
"GSM4286835","GSM4286836","GSM4286837","GSM4286838","GSM4286839","GSM4286840","GSM4286841","GSM4286842","GSM4286843","GSM4286844","GSM4286845","GSM4286846","GSM4286847","GSM4286848","GSM4286849","GSM4286850","GSM4286851","GSM4286852","GSM4286853","GSM4286854","GSM4286855","GSM4286856","GSM4286857","GSM4286858","GSM4319659","GSM4319660","GSM4319665","GSM4319666","GSM4319673","GSM4319674",
"GSM4319682","GSM4368433","GSM4368434","GSM4387501","GSM4387502","GSM4387503","GSM4387504","GSM4387505","GSM4387506","GSM4387507","GSM4387508","GSM4387509","GSM4387510","GSM4387511","GSM4387512","GSM4387513","GSM4387514","GSM4387515","GSM4387516","GSM4387517","GSM4387518","GSM4387519","GSM4387520","GSM4387521","GSM4387522","GSM4387523","GSM4387524","GSM4387525","GSM4387526","GSM4387527",
"GSM4387528","GSM4387529","GSM4387530","GSM4387531","GSM4387532","GSM4387533","GSM4387534","GSM4387535","GSM4387536","GSM4387537","GSM4387538","GSM4387632","GSM4387633","GSM4387634","GSM4387635","GSM4387637","GSM4387638","GSM4387639","GSM4387640","GSM4387644","GSM4387645","GSM4387646","GSM4387647","GSM4387750","GSM4387751","GSM4387752","GSM4387753","GSM4387754","GSM4387755","GSM4387756",
"GSM4387757","GSM4387758","GSM4387759","GSM4387760","GSM4387761","GSM4387762","GSM4387763","GSM4387764","GSM4387765","GSM4387766","GSM4387768","GSM4387769","GSM4387770","GSM4387771","GSM4387772","GSM4387773","GSM4387774","GSM4387775","GSM4387776","GSM4387777","GSM4387778","GSM4387779","GSM4387780","GSM4387781","GSM4387782","GSM4387783","GSM4387784","GSM4387785","GSM4387870","GSM4387871",
"GSM4387872","GSM4387873","GSM4387874","GSM4387875","GSM4387876","GSM4387877","GSM4387878","GSM4387879","GSM4387880","GSM4387881","GSM4387882","GSM4387883","GSM4387884","GSM4387885","GSM4387886","GSM4387887","GSM4387888","GSM4387889","GSM4387890","GSM4387891","GSM4387963","GSM4387964","GSM4387965","GSM4387966","GSM4387967","GSM4387968","GSM4387969","GSM4387970","GSM4387971","GSM4387972",
"GSM4387973","GSM4387974","GSM4387975","GSM4387976","GSM4387977","GSM4387978","GSM4387979","GSM4387980","GSM4387981","GSM4387982","GSM4387983","GSM4387984","GSM4387985","GSM4387986","GSM4388033","GSM4388034","GSM4388035","GSM4388036","GSM4388037","GSM4388038","GSM4388039","GSM4388040","GSM4388041","GSM4388042","GSM4388043","GSM4388044","GSM4388045","GSM4388046","GSM4388047","GSM4388048",
"GSM4388049","GSM4388050","GSM4388051","GSM4388052","GSM4388053","GSM4388054","GSM4388055","GSM4388056","GSM4388057","GSM4388058","GSM4388059","GSM4431475","GSM4431476","GSM4431477","GSM4431478","GSM4431479","GSM4431480","GSM4431481","GSM4431482","GSM4431483","GSM4431484","GSM4522895","GSM4522896","GSM4522901","GSM4522911","GSM4522919","GSM4522922","GSM4522924","GSM4522934","GSM4594410",
"GSM4594411","GSM4594412","GSM4594413","GSM4722036","GSM4722037","GSM4722038","GSM4722039","GSM4722040","GSM4722041","GSM4722042","GSM4722043","GSM4722044","GSM4722045","GSM4722046","GSM4722047","GSM4749275","GSM4762820","GSM4762821","GSM4762823","GSM4762825","GSM4762829","GSM3532110","GSM3532111","GSM3532112","GSM3532113","GSM3532114","GSM3532115","GSM3891066","GSM4036966","GSM4036967",
"GSM4036968","GSM4036969","GSM4036970","GSM4036971","GSM4036972","GSM4036973","GSM4036974","GSM4036975","GSM4036976","GSM4036977","GSM4036978","GSM4036979","GSM4036980","GSM4036981","GSM4036982","GSM4036983","GSM4036984","GSM4036985","GSM4036986","GSM4036987","GSM4036988","GSM4036989","GSM4142998","GSM4142999","GSM4143000","GSM4143001","GSM4143002","GSM4143003","GSM4143004","GSM4143005",
"GSM4143006","GSM4143007","GSM4143008","GSM4143009","GSM4143010","GSM4143011","GSM4143012","GSM4143013","GSM4143014","GSM4143015","GSM4143016","GSM4143017","GSM4143018","GSM4143019","GSM4143020","GSM4143021","GSM4143022","GSM4143023","GSM4143024","GSM4143025","GSM4143026","GSM4143027","GSM4143028","GSM4143029","GSM4143030","GSM4143031","GSM4143032","GSM4143033","GSM4143034","GSM4143035",
"GSM4143036","GSM4143037","GSM4143038","GSM4143039","GSM4143040","GSM4143041","GSM4143042","GSM4143044","GSM4143045","GSM4143046","GSM4143047","GSM4143048","GSM4143049","GSM4143050","GSM4143051","GSM4143052","GSM4143053","GSM4143054","GSM4143055","GSM4143056","GSM4143057","GSM4143058","GSM4143059","GSM4143060","GSM4143061","GSM4143062","GSM4182426","GSM4182427","GSM4182428","GSM4182429",
"GSM4182430","GSM4182431","GSM4182432","GSM4182433","GSM4182434","GSM4182435","GSM4190560","GSM4200846","GSM4201098","GSM4201099","GSM4201100","GSM4201101","GSM4201533","GSM4201534","GSM4201535","GSM4201536","GSM4263860","GSM4263861","GSM4263862","GSM4263863","GSM4263864","GSM4263865","GSM4263866","GSM4263867","GSM4263868","GSM4263869","GSM4263870","GSM4263871","GSM4368678","GSM4368679",
"GSM4368680","GSM4368681","GSM4387354","GSM4387355","GSM4387356","GSM4387357","GSM4387358","GSM4387359","GSM4387360","GSM4387361","GSM4387362","GSM4387363","GSM4387364","GSM4387365","GSM4387366","GSM4387367","GSM4387368","GSM4387369","GSM4387370","GSM4387371","GSM4387372","GSM4387373","GSM4387374","GSM4387375","GSM4387376","GSM4387377","GSM4387636","GSM4387641","GSM4387642","GSM4387643",
"GSM4387648","GSM4387649","GSM4387650","GSM4387651","GSM4387652","GSM4387653","GSM4387654","GSM4387655","GSM4387767","GSM4387869","GSM4431985","GSM4431986","GSM4431987","GSM4431990","GSM4431991","GSM4431992","GSM4431993","GSM4431995","GSM4431996","GSM4431997","GSM4431998","GSM4431999","GSM4432000","GSM4432001","GSM4432003","GSM4432004","GSM4432005","GSM4432006","GSM4432007","GSM4432008",
"GSM4474536","GSM4474537","GSM4474538","GSM4474539","GSM4474540","GSM4474541","GSM4548078","GSM4548079","GSM4548080","GSM4548081","GSM4610605","GSM4670657","GSM4670658","GSM4670659","GSM4670660","GSM4670661","GSM4670662","GSM4705014","GSM4705015","GSM4775703","GSM4801173","GSM4801174","GSM4801175","GSM4801176","GSM4801177","GSM4801178","GSM4801179","GSM4801180","GSM4801181","GSM4801182",
"GSM4801183","GSM4801184","GSM4801185","GSM4801186","GSM4801187","GSM4801188","GSM4801189","GSM4801190","GSM4801191","GSM4801192","GSM4801193","GSM4801194","GSM4801195","GSM4801197","GSM4801198","GSM4801199","GSM4801200","GSM4801201","GSM4801202","GSM4801203","GSM4801204","GSM4801205","GSM4801206","GSM4801207","GSM4801208","GSM4801209","GSM4801210","GSM4801211","GSM4801212","GSM4811589",
"GSM4811590","GSM2095860","GSM2095861","GSM2095862","GSM2095863","GSM2095864","GSM2095865","GSM2095866","GSM2095867","GSM3852960","GSM3852961","GSM3852962","GSM3852964","GSM3852965","GSM3852966","GSM3852967","GSM3852968","GSM3852969","GSM3852970","GSM3852971","GSM3852972","GSM3852973","GSM3852974","GSM3852975","GSM3852976","GSM3852977","GSM3852978","GSM3852979","GSM3852980","GSM3852981",
"GSM3852982","GSM3852983","GSM4099468","GSM4099469","GSM4099470","GSM4099471","GSM4378216","GSM4378217","GSM4378218","GSM4378219","GSM4378220","GSM4378221","GSM4378222","GSM4378223","GSM4378224","GSM4378225","GSM4378226","GSM4378227","GSM4378228","GSM4378229","GSM4378230","GSM4378231","GSM4644237","GSM4698737","GSM4698739","GSM4765003","GSM4765004","GSM4765005","GSM4765006","GSM4765007",
"GSM4765008","GSM4765009","GSM4765010","GSM4765011","GSM4765012","GSM4800650","GSM4800652","GSM4800654","GSM4800655","GSM4800657","GSM4800658","GSM4800659","GSM4802547","GSM4802548","GSM4802560","GSM4802651","GSM4802652","GSM4802653","GSM4802654","GSM4802655","GSM4802656","GSM4802658","GSM4802659","GSM4802660","GSM4802661","GSM4802662","GSM4802665","GSM4802666","GSM4802667","GSM4802668",
"GSM4802669","GSM4802670","GSM4802671","GSM4802672","GSM4914174","GSM4914175","GSM4914176","GSM4914177","GSM4914178","GSM4995619","GSM4995620","GSM4995622","GSM4995623","GSM4995624","GSM4995625","GSM4995626","GSM4995627","GSM4995628","GSM4995629","GSM4995630","GSM5029338","GSM5069014","GSM5069015","GSM5069020","GSM5069021","GSM5077858","GSM5093491","GSM5093492","GSM5093493","GSM5093494",
"GSM5093495","GSM5093496","GSM5093499","GSM5093501","GSM5093502","GSM5093503","GSM5093504","GSM5093505","GSM5093508","GSM5093509","GSM5151397","GSM5151399","GSM5151400","GSM5203203","GSM5203204","GSM5203205","GSM5203206","GSM5239436","GSM5239437","GSM5239438","GSM5260049","GSM5260061","GSM5260062","GSM5260068","GSM5260074","GSM5260079","GSM5260092","GSM5260102","GSM5260129","GSM5260136",
"GSM5260148","GSM5260149","GSM5356230","GSM5356234","GSM5356238","GSM5356242","GSM5356246","GSM5356254","GSM5356258","GSM5356262","GSM5356266","GSM5356270","GSM5356274","GSM5356278","GSM5356282","GSM5356286","GSM5356290","GSM5356294","GSM5356298","GSM5356302","GSM5356306","GSM5356310","GSM5356314","GSM5356318","GSM5356322","GSM5356326","GSM5356329","GSM5356333","GSM5356337","GSM5356341",
"GSM5356345","GSM5356349","GSM5356353","GSM5356357","GSM5356361","GSM5356365","GSM5356368","GSM5387904","GSM5387905","GSM5387906","GSM5387907","GSM5387908","GSM5387909","GSM5387910","GSM5387911","GSM5471539","GSM5471540","GSM5471541","GSM5471542","GSM5471543","GSM5471544","GSM5471545","GSM5471546","GSM5471547","GSM5471548","GSM5471549","GSM5471550","GSM5471569","GSM5471570","GSM5471571",
"GSM5471572","GSM5471573","GSM5471574","GSM5471575","GSM5471577","GSM5471578","GSM5471579","GSM5471580","GSM5555527","GSM5555528","GSM5555529","GSM5555530","GSM5555531","GSM5555532","GSM5555533","GSM5555534","GSM5555536","GSM5555537","GSM5555538","GSM5555539","GSM5555540","GSM5555541","GSM5555542"])

sample_meta.to_csv("mouse_blood_metadata.csv", sep = "|")