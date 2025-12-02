import xml.etree.ElementTree as ET

can_run_bf16_ops = {"Convolution", "GroupConvolution", "FullyConnected", "RNNCell", "RNNSeq",
                    "MatMul", "MulAdd", "Add", "ROIPooling", "Interpolate", "AvgPool",
                    "MVN", "MaxPool", "Eltwise"}

can_run_bf16_ops = {"Convolution", "GroupConvolution", "FullyConnected", "RNNCell", "RNNSeq",
                    "MatMul", "MulAdd", "Add", "ROIPooling", "Interpolate", "AvgPool",
                    "MVN", "MaxPool", "Eltwise"}


class ModelCreator():
    def __init__(self, xml_path):
        self.doc = ET.parse(xml_path)
        self.root = self.doc.getroot()
        self.layers = self.root.find("layers")

    def print_layer(self):
        for layer in self.layers.findall("layer"):
            name = layer.attrib.get("name")
            ltype = layer.attrib.get("type")
            if ltype in can_run_bf16_ops:
                print(f"Layer name={name}, type={ltype}")

    def search_layer(self, layer_name):
        for layer in self.layers.findall("layer"):
            name = layer.attrib.get("name")
            if name == layer_name:
                return layer
        return None

			# <rt_info>
		    # 	<attribute name="precise" version="0" value="true" />
			# </rt_info>
    def update_all(self, new_xml_path):
        for layer in self.layers.findall("layer"):
            name = layer.attrib.get("name")
            ltype = layer.attrib.get("type")
            if ltype in can_run_bf16_ops or True:
                rt_info = layer.find("rt_info")
                if rt_info is None:
                    rt_info = ET.Element("rt_info")
                    layer.append(rt_info)
                self.update_rt_info(rt_info, True)
        self.doc.write(new_xml_path)

    def update_rt_info(self, rt_info, force_fp32: bool):
        rt_info_attr = None
        for child in rt_info.iter("attribute"):
            if "precise" == child.attr.get("name"):
                rt_info_attr = child
                break
        force_fp32_str = "false"
        if force_fp32:
            force_fp32_str = "true"
        if rt_info_attr is None:
            rt_info_attr = ET.Element(
                "attribute", {"name": "precise", "version": "0"})
            rt_info.append(rt_info_attr)
        else:
            rt_info_attr.set("value", force_fp32_str)

    def clearup_force_fp32(self):
        for layer in self.layers.iter("layer"):
            rt_info = layer.find("rt_info")
            if rt_info is not None:
                for child in rt_info.iter("attribute"):
                    if "precise" == child.get("name"):
                        rt_info.remove(child)

    def create_new_xml(self, new_xml_path, force_fp32_set):
        self.clearup_force_fp32()
        for iname in force_fp32_set:
            layer = self.search_layer(iname)
            name = layer.get("name")
            rt_info = layer.find("rt_info")
            if rt_info is None:
                rt_info = ET.Element("rt_info")
                layer.append(rt_info)
            self.update_rt_info(rt_info, True)
        self.doc.write(new_xml_path)

input_file  = "pretrained_models/FireRedASR-AED-L/ov_model/FireRedASR_AED_decoder0_ov.xml.bak"
output_file = "pretrained_models/FireRedASR-AED-L/ov_model/FireRedASR_AED_decoder0_ov.xml"

model = ModelCreator(input_file)

model.update_all(output_file)
print(f"export output_file: {output_file} successfully.")


input_file  = "pretrained_models/FireRedASR-AED-L/ov_model/FireRedASR_AED_decoder1_ov.xml.bak"
output_file = "pretrained_models/FireRedASR-AED-L/ov_model/FireRedASR_AED_decoder1_ov.xml"

model = ModelCreator(input_file)

model.update_all(output_file)
print(f"export output_file: {output_file} successfully.")


input_file  = "pretrained_models/FireRedASR-AED-L/ov_model/FireRedASR_AED_encoder_ov.xml.bak"
output_file = "pretrained_models/FireRedASR-AED-L/ov_model/FireRedASR_AED_encoder_ov.xml"

model = ModelCreator(input_file)

model.update_all(output_file)
print(f"export output_file: {output_file} successfully.")

