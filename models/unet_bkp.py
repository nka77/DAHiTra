def forward(self, x1, x2):

        # Encoder 1
        x_1 = x1
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)
        enc4_1 = self.conv4(enc3_1)
        enc5_1 = self.conv5(enc4_1)

        # Encoder 2
        x_2 = x2
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)
        enc4_2 = self.conv4(enc3_2)
        enc5_2 = self.conv5(enc4_2)

        # Bottleneck
        enc5_1 = (self.ca_bottle_max(enc5_1)*enc5_1)
        enc5_2 = (self.ca_bottle_max(enc5_2)*enc5_2)

        # try dot product as attention??
        # enc5 = torch.einsum('bcij,bcij->bcij' , enc5_1, enc5_2)
        # enc5 = self.ca_bottle_avg_min(enc5)*enc5

        enc5 = self.ca_skip_5(enc5_1,enc5_2)

        B_, C_, H_, W_ = enc5.shape
        enc5_i = enc5.view([B_, C_, H_*W_])
        enc5_i = self.transformer(enc5_i)
        # print(enc5.shape)
        enc5_i = enc5_i.view([B_, C_, H_, W_])
        enc5 = self.ca_skip_5(enc5_i,enc5)

        # Decoder
        enc4 = self.ca_skip_4(enc4_1, enc4_2)
        # enc4 = attention_block(enc4_1, enc4_2, self.encoder_filters[-2])
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        enc3 = self.ca_skip_3(enc3_1, enc3_2)
        # enc3 = attention_block(enc3_1, enc3_2, self.encoder_filters[-3])
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        enc2 = self.ca_skip_2(enc2_1, enc2_2)
        # enc2 = attention_block(enc2_1, enc2_2, self.encoder_filters[-4])
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        enc1 = self.ca_skip_2(enc1_1, enc1_2)
        # enc1 = attention_block(enc1_1, enc1_2, self.encoder_filters[-5])
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        out = self.res(dec10)
        return out
