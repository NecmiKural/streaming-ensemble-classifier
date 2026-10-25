"""Build sizeof-agent.jar, which loads MOA's SizeOfAgent so measureByteSize() works.

Without the agent, MOA's measureByteSize() returns 0, so a HoeffdingTree never
sees itself grow and max_byte_size is never enforced. Usage:

    python make_sizeof_agent.py
    export JAVA_TOOL_OPTIONS="-javaagent:$PWD/sizeof-agent.jar"
"""

import os
import zipfile

import capymoa

moa_jar = os.path.join(os.path.dirname(capymoa.__file__), "jar", "moa.jar")
manifest = (
    "Manifest-Version: 1.0\n"
    "Premain-Class: sizeof.agent.SizeOfAgent\n"
    f"Class-Path: file:{moa_jar}\n\n"
)
with zipfile.ZipFile("sizeof-agent.jar", "w") as jar:
    jar.writestr("META-INF/MANIFEST.MF", manifest)
print(f"wrote sizeof-agent.jar (agent class from {moa_jar})")
