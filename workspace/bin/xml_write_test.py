import argparse


parser = argparse.ArgumentParser(description='collect each test coverage.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')

args = parser.parse_args()
# args
# BUILD_dir
##

path = "/tmp/" + args.project_id + "/" + args.bug_id + "/build.xml"  ##Math,Closure,Lang
#path = "/tmp/" + args.project_id + "/" + args.bug_id + "/ant/build.xml"   ##Chart

import xml.etree.ElementTree as ET

tree = ET.parse(path)
root = tree.getroot()

#'''   ###Math, Lang
for child in root:
    if "id" in child.attrib and (child.attrib["id"] == "compile.classpath" \
        or child.attrib["id"] == "test.classpath" or child.attrib["id"] == "build.classpath"):
        cloverpath = ET.SubElement(child, "pathelement")
        cloverpath.set("location", "/root/clover/lib/clover.jar")
        print(child.attrib)
#'''
'''   ###Lang at Version21 over
for child in root:
    if "id" in child.attrib and (child.attrib["id"] == "compile.classpath" \
        or child.attrib["id"] == "test.classpath" or child.attrib["id"] \
              == "build.test.classpath"):
        cloverpath = ET.SubElement(child, "pathelement")
        cloverpath.set("location", "/root/clover/lib/clover.jar")
        print(child.attrib)
'''
'''   ###Chart
for child in root:
    if child.attrib['name'] == "initialise":
        for c_child in child:
            if "id" in c_child.attrib and c_child.attrib['id'] ==\
                  "build.classpath":
                cloverpath = ET.SubElement(c_child, "pathelement")
                cloverpath.set("location", "/root/clover/lib/clover.jar")
                print(c_child.attrib)
'''
'''
for child in root:
    if "id" in child.attrib and (child.attrib["id"] == "srcclasspath.path" \
        or child.attrib["id"] == "allclasspath.path"):
        cloverpath = ET.SubElement(child, "pathelement")
        cloverpath.set("location", "/root/clover/lib/clover.jar")
        print(child.attrib)
    
    #cloverpath = ET.SubElement(child, "pathelement")
    #cloverpath.set("location", "/root/clover/lib/clover.jar")
    #print(child.attrib)
'''


tree.write(path)