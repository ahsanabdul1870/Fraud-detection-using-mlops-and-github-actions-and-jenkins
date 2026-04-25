from kfp import dsl
from kfp import kubernetes

@dsl.component
def my_comp():
    pass

@dsl.pipeline(name="test")
def my_pipeline():
    task = my_comp()
    kubernetes.mount_pvc(task, pvc_name="my-pvc", mount_path="/mnt")

from kfp import compiler
compiler.Compiler().compile(my_pipeline, "/tmp/test.yaml")
