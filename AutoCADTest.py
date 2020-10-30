from pyautocad import Autocad, APoint


acad = Autocad(create_if_not_exists=True, visible=False)

# 打开文件
filepath = r'C:\Users\Administrator\Desktop\dwgFiles\Circle.dwg'
filename = filepath.split('\\')[-1]
acad.Application.Documents.Open(filepath)

# 设定文件为当前
acad.Application.Documents(filename).Activate()

# 遍历文件中的对象
for obj in acad.iter_objects():
    print(obj.ObjectName)
    if obj.ObjectName == 'AcDbCircle':
        desc = u"Shape:Circle\n"
        desc += u"Center:(" + str(obj.Center[0]) + u"," + str(obj.Center[1]) + u")\n"
        desc += u"Raidius:" + str(obj.Radius) + u"\n"
        desc += u"Diameter:" + str(obj.Diameter) + u"\n"
        desc += u"Area:" + str(obj.Area) + u"\n"
        desc += u"Circumference:" + str(obj.Circumference)
        px = obj.Center[0] + 1.1 * obj.Radius
        py = obj.Center[1] + 1.1 * obj.Radius
        desc_text = acad.model.addMText(APoint(px, py), obj.Radius, desc)
        h = obj.Radius / 7
        desc_text.Height = h
        desc_text.Update()
    elif obj.ObjectName == 'AcDbText':
        print(obj.TextString)


# 设置绘图仪管理器的工作目录
ACADPref = acad.Application.preferences.Files

ACADPref.PrinterConfigPath = r"C:\Users\Administrator\AppData\Roaming\Autodesk\AutoCAD 2020\R23.1\chs\Plotters"

oplot = acad.ActiveDocument.PlotConfigurations.Add("PNG", acad.ActiveDocument.ActiveLayout.ModelType)

acad.ActiveDocument.ActiveLayout.ConfigName = "PublishToWeb PNG.pc3"
acad.ActiveDocument.SetVariable("Filedia", 0)

acad.ActiveDocument.SetVariable("BACKGROUNDPLOT", 0)
acad.ActiveDocument.Plot.QuietErrorMode = True

acad.ActiveDocument.Plot.PlotToFile("D:\\" + "test" + ".png")

oplot.Delete()
oplot = None

# 关闭文档并不保存
acad.ActiveDocument.Close(False)
# 退出程序
acad.Application.Quit()


