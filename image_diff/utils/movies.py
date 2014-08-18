    def make_movie(self, movie_obj):
        '''
        TO CREATE MOVIE
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        '''
        
        # Define the codec and create VideoWriter object
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        #fourcc = cv2.cv.CV_FOURCC('M','P','V','4')
        #fourcc = cv2.cv.CV_FOURCC('M','P','E','G')
        fourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
        out = cv2.VideoWriter(self.video_path + '.mp4',fourcc, self.fps,
                              (self.width/4,self.height/4))
        print "Making a movie, I am rank: ", self.rank
        for i in range(len(self.frame_no)):
            if i < len(self.frame_no):
            
                if i == len(self.frame_no) and self.frame_chi[i] > 3.:
                    print "Oh be a fine girl kiss me now"
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] > 3.:
                    print "it's true", self.frame_no[i], self.rank, len(self.frame_no)

                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        out.write(frame)
                elif self.frame_chi[i] > 3. and self.frame_chi[i+1] < 3.:
                    print "not true", self.frame_no[i], self.rank, len(self.frame_no)
                
                    for j in range(int(self.frame_no[i]), int(self.frame_no[i+1]), 1):
                        self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, j)
                        success, frame = self.cap.read()
                        out.write(frame)
                        
