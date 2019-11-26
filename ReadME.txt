3D GAN  sample project 
To Run this Projects

1.Please download the 3D shapes from this location http://3dshapenets.cs.princeton.edu/3D
2.On unzipping the shapes you will find many different object shapes under \volumetric_data 
3. Make the following changes in the 3DGan.py
  a.Change the object_name of your choice found in the volumetric_data
  b.Change the location to file on your device;
  c.Make an results directory for storing of fake images (location same as 3DGan.py)
    def load_data(self):
        self.Initialize()
        object_name = "airplane" # a.Change the object of your choice found in the volumetric_data directory
        data_dir = "E:\\workdirectory\\Code Name Val Halen\\DS Sup\\DL\\GAN - Projects\\9781789136678_Code\\Chapter02\\3DShapeNets\\volumetric_data" \
               "\\{}\\30\\train\\*.mat".format(object_name) # b.Change the location to file on your device;
        print(data_dir)
        volumes = self.get3DImages(data_dir=data_dir)
        self.volumes = volumes[..., np.newaxis].astype(np.float)
        print("No of volumes")
        print(len(self.volumes))
        print(self.volumes.shape[0])
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
        self.Create_Generator()
        self.Create_Discriminator()
        tensorboard.set_model(self.gen_model)
        tensorboard.set_model(self.dis_model)
        self.tensorboard = tensorboard
4. The code can be executed as is , the fake images will be stored results
