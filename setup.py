from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil

class LocalSetup(install):
    def run(self):
        install.run(self)

        # if unix
        if os.name == 'posix':
            self.install_linux()
        # if windows
        elif os.name == 'nt':
            self.install_windows()
        else:
            print(f'OS {os.name} not supported')

    def install_linux(self):
        stepNames = ['TNT_1_Prediction', 'TNT_2_CreateMask', 'TNT_3_ApplyMask']

        for stepName in stepNames:
            # Paths for the desktop file and icon
            icon_file_source = os.path.join(os.getcwd(), f'setup/{stepName}.png')
            assert os.path.exists(icon_file_source), f'{icon_file_source} does not exist'

            if not os.path.exists(os.path.expanduser('~/.local/share/icons/')):
                os.makedirs(os.path.expanduser('~/.local/share/icons/'))

            if not os.path.exists(os.path.expanduser('~/.local/share/applications/')):
                os.makedirs(os.path.expanduser('~/.local/share/applications/'))

            icon_file_dest = os.path.expanduser(f'~/.local/share/icons/{stepName}.png')
            desktop_file_dest = os.path.expanduser(f'~/.local/share/applications/{stepName}.desktop')

            conda_executable_path = shutil.which('conda')

            desktop_file_content = f"""[Desktop Entry]
Version=0.1
Type=Application
Name={stepName}
Comment=Run TNTAnalysis Prediction GUI
Exec={conda_executable_path} run -n TNTAnalysis {stepName}
Icon={icon_file_dest}
Terminal=true
Categories=Utility;"""

            print(f'Writing desktop file to {desktop_file_dest}')
            with open(desktop_file_dest, 'w') as desktop_file:
                desktop_file.write(desktop_file_content)


            print(f'Copying {icon_file_source} to {icon_file_dest}')
            shutil.copyfile(icon_file_source, icon_file_dest)

            print(f'Setting executable permissions for {desktop_file_dest}')
            os.chmod(desktop_file_dest, 0o755)

    def convert_png_to_ico(png_path, ico_path):
        from PIL import Image
        img = Image.open(png_path)
        img.save(ico_path, format='ICO')

    def install_windows(self):
        import win32com.client
        from PIL import Image


        stepNames = ['TNT_1_Prediction', 'TNT_2_CreateMask', 'TNT_3_ApplyMask']

        for stepName in stepNames:
            icon_file_source = os.path.join(os.getcwd(), f'setup/{stepName}.png')
            assert os.path.exists(icon_file_source), f'{icon_file_source} does not exist'

            icon_file_dest = icon_file_source.replace(".png", ".ico")
            print(f'Converting {icon_file_source} to {icon_file_dest}')
            img = Image.open(icon_file_source)
            img.save(icon_file_dest, format='ICO')


            export_dir = os.path.join('.')
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            bat_file_dest = os.path.join(os.getcwd(), f'setup/{stepName}.bat')

            bat_file_content = (f"call activate TNTAnalysis\n"
                                f"{stepName}.exe\n"
                                f"pause"
                                )

            print(f'Writing batch file to {bat_file_dest}')
            with open(bat_file_dest, 'w') as bat_file:
                bat_file.write(bat_file_content)

            shortcut_dest = os.path.join(os.getcwd(), f'{stepName}.lnk')
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortcut(shortcut_dest)
            shortcut.TargetPath = bat_file_dest
            shortcut.WorkingDirectory = os.getcwd()
            shortcut.IconLocation = icon_file_dest
            shortcut.save()


if __name__ == '__main__':
    setup(
        name='TNTAnalysis',
        version='0.1',
        packages=find_packages(),
        cmdclass={
            'install': LocalSetup,
        },
        entry_points={
            'console_scripts': [
                'TNT_1_Prediction = TNTAnalysis.A_GUI_lif:run_GUI',
                'TNT_2_CreateMask = TNTAnalysis.B_CreateMask:run_GUI',
                'TNT_3_ApplyMask = TNTAnalysis.C_ApplyMask:run_GUI',
            ],
        },
    )
