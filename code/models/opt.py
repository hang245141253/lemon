import paddlelite.lite as lite

a=lite.Opt()
# 非combined形式
# a.set_model_dir("/home/firefly/Paddle-Lite-2.8-rc/lite/demo/python/mobilenet_v1")

# conmbined形式
a.set_model_file("./Hapi_MyCNN.pdmodel")
a.set_param_file("./Hapi_MyCNN.pdiparams")

a.set_optimize_out("model")
a.set_valid_places("arm")

a.run()