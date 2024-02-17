import org.junit.runner.JUnitCore;
import org.junit.runner.Request;
import org.junit.runner.Result;
import java.util.List;
import org.junit.runner.notification.Failure;
import junit.runner.Version;


public class SingleJUnitTestRunner {
    public static void main(String... args) throws ClassNotFoundException {
        // System.out.println(junit.runner.Version.id());
        String[] classAndMethod = args[0].split("#");
        Request request = Request.method(Class.forName(classAndMethod[0]),
                classAndMethod[1]);

        Result result = new JUnitCore().run(request);
        List<Failure> errList = result.getFailures();
        for (Failure fail: errList){
            System.out.println(fail.getException());
            System.out.println(fail.getTrace());
        }

        System.out.println(result.wasSuccessful() ? "successed" : "failed");
        System.exit(result.wasSuccessful() ? 0 : 1);
    }
}